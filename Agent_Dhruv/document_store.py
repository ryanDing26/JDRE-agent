# document_store.py
import os, re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from functools import lru_cache

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import faiss
    _FAISS_OK = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_OK = False

from sentence_transformers import SentenceTransformer

from rag_config import (
    EMBED_MODEL_NAMES, DEVICE, USE_COSINE,
    FAISS_USE_GPU, FAISS_GPU_ID, EMBED_BATCH_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_WARN
)

def _pick_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

_device = _pick_device(DEVICE)

def clean_text(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(t: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [t.strip()]
    if overlap >= size:
        overlap = max(0, size - 1)
    out = []
    stride = max(1, size - overlap)
    t = t.strip()
    n = len(t); i = 0
    while i < n:
        out.append(t[i:i+size])
        i += stride
    return out

@dataclass
class Chunk:
    id: int
    hadm_id: Any
    text: str
    context: str

class MultiModelIndex:
    """
    Holds multiple SentenceTransformer encoders + one FAISS index per model.
    Primary embeddings are stored for MMR and introspection; others optional.
    """
    def __init__(self, model_names: List[str], device: str):
        assert len(model_names) >= 1, "At least one embedding model is required."
        self.device = device
        self.model_names: List[str] = [m.strip() for m in model_names if m.strip()]
        self.models: Dict[str, SentenceTransformer] = {
            name: SentenceTransformer(name, device=self.device)
            for name in self.model_names
        }
        self.indices: Dict[str, Any] = {}  # faiss.Index or GPU proxy
        self.embeddings_primary: Optional[np.ndarray] = None
        self.dim_by_model: Dict[str, int] = {}
        self.use_cosine = USE_COSINE
        self._faiss_gpu = False

    def _ensure_faiss(self):
        if not _FAISS_OK:
            raise RuntimeError(
                "faiss is not installed. Install faiss-cpu (or faiss-gpu via conda)."
            )

    def _maybe_gpu_index(self, cpu_index):
        if not FAISS_USE_GPU:
            return cpu_index
        self._ensure_faiss()
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, FAISS_GPU_ID, cpu_index)
        except Exception:
            return cpu_index

    def _encode_model(self, model_name: str, texts: List[str]) -> np.ndarray:
        model = self.models[model_name]
        embs = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=self.use_cosine,
            show_progress_bar=False,
        )
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)
        return embs

    def build(self, texts: List[str]):
        self._ensure_faiss()
        # Build index for each model
        for i, name in enumerate(self.model_names):
            embs = self._encode_model(name, texts)
            dim = int(embs.shape[1])
            self.dim_by_model[name] = dim
            if self.use_cosine:
                index_cpu = faiss.IndexFlatIP(dim)
            else:
                index_cpu = faiss.IndexFlatL2(dim)
            index = self._maybe_gpu_index(index_cpu)
            index.add(embs)
            self.indices[name] = index
            if i == 0:
                self.embeddings_primary = embs
        # best-effort check
        self._faiss_gpu = any(
            getattr(self.indices[n], "is_trained", True) and "Gpu" in type(self.indices[n]).__name__
            for n in self.indices
        )

    def encode_query_multi(self, query: str) -> Dict[str, np.ndarray]:
        return {
            name: self.models[name].encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=self.use_cosine
            )[0].astype(np.float32, copy=False)
            for name in self.model_names
        }

    def search_multi(self, qvecs: Dict[str, np.ndarray], top_k: int) -> Dict[int, float]:
        """
        Search each model-specific index and merge by normalized score.
        Returns dict: doc_idx -> aggregated_score (0..1).
        """
        agg: Dict[int, float] = {}
        for name, qv in qvecs.items():
            index = self.indices[name]
            D, I = index.search(qv.reshape(1, -1).astype(np.float32), top_k)
            drow, irow = D[0], I[0]
            for score_raw, idx in zip(drow, irow):
                if idx == -1:
                    continue
                # normalize inner product/cosine [-1,1] -> [0,1]
                if self.use_cosine:
                    score = max(0.0, min(1.0, 0.5 * (float(score_raw) + 1.0)))
                else:
                    # For L2, convert to a decreasing positive score (rough heuristic)
                    score = 1.0 / (1.0 + float(score_raw))
                agg[idx] = agg.get(idx, 0.0) + score
        # average over models to keep consistent scale
        if len(qvecs) > 0:
            for k in list(agg.keys()):
                agg[k] /= float(len(qvecs))
        return agg

class DocumentStore:
    """
    In-memory store for chunks + multi-model indices.
    """
    def __init__(self):
        names = [m.strip() for m in EMBED_MODEL_NAMES.split(",") if m.strip()]
        self.mm: MultiModelIndex = MultiModelIndex(names, _device)
        self.chunks: List[Chunk] = []
        self._id2row: Dict[int, Chunk] = {}
        self.device = _device

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "TEXT",
        context_col: Optional[str] = "CONTEXT",
        id_col: Optional[str] = "HADM_ID",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ) -> Dict[str, Any]:

        # Column normalization
        cols = {c.lower(): c for c in df.columns}
        def pick(col_name: str, fallback: Optional[str] = None):
            if col_name.lower() in cols:
                return cols[col_name.lower()]
            return fallback

        text_col = pick(text_col, fallback=list(df.columns)[0])
        context_col = pick(context_col, fallback=None)
        id_col = pick(id_col, fallback=None)

        # Build chunk list
        chunks: List[Chunk] = []
        next_id = 0
        for _, row in df.iterrows():
            text = clean_text(str(row.get(text_col, "")))
            if not text:
                continue
            hadm = row.get(id_col, None) if id_col else None
            ctx = clean_text(str(row.get(context_col, ""))) if context_col else ""
            parts = chunk_text(text, chunk_size, chunk_overlap)
            for part in parts:
                chunks.append(Chunk(id=next_id, hadm_id=hadm, text=part, context=ctx))
                next_id += 1

        if len(chunks) > MAX_CHUNKS_WARN:
            # prevent accidental OOM on giant files
            raise RuntimeError(
                f"Too many chunks ({len(chunks)}). Increase CHUNK_SIZE or reduce input."
            )

        self.chunks = chunks
        self._id2row = {c.id: c for c in chunks}

        texts = [c.text for c in chunks]
        if not texts:
            raise RuntimeError("No text found to index. Check your column names.")

        # Build multi-model FAISS indices
        self.mm.build(texts)

        return {
            "num_rows": len(df),
            "num_chunks": len(chunks),
            "embedding_dims": self.mm.dim_by_model,
            "faiss_gpu": self.mm._faiss_gpu,
            "device": self.device,
            "models": self.mm.model_names,
        }

    def primary_vec(self, idx: int) -> Optional[np.ndarray]:
        embs = self.mm.embeddings_primary
        return None if embs is None else embs[int(idx)]

    def get_chunk(self, idx: int) -> Chunk:
        return self._id2row[int(idx)]

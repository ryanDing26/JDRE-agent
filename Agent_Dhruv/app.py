# app.py
import io
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from schemas import LoadJSONRequest, QueryRequest, QueryResponse, EvidenceItem
from document_store import DocumentStore
from retrieval import rerank_hybrid
from generation import answer_with_context, reflect_and_refine
from rag_config import CHUNK_SIZE, CHUNK_OVERLAP

app = FastAPI(title="Biomedical RAG API", version="1.0.0")

STORE = DocumentStore()
INDEX_READY = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": STORE.device,
        "models": STORE.mm.model_names,
    }

@app.post("/load_json")
def load_json(req: LoadJSONRequest):
    global INDEX_READY
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must be non-empty")
    df = pd.DataFrame(req.rows)
    stats = STORE.load_from_dataframe(
        df,
        text_col=req.text_col,
        context_col=req.context_col,
        id_col=req.id_col,
        chunk_size=req.chunk_size or CHUNK_SIZE,
        chunk_overlap=req.chunk_overlap or CHUNK_OVERLAP
    )
    INDEX_READY = True
    return {"ok": True, "stats": stats}

@app.post("/load_csv")
def load_csv(file: UploadFile = File(...),
             text_col: str = "TEXT",
             context_col: str = "CONTEXT",
             id_col: str = "HADM_ID",
             chunk_size: int = CHUNK_SIZE,
             chunk_overlap: int = CHUNK_OVERLAP):
    global INDEX_READY
    if not file.filename.endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Only .csv or .tsv files supported.")
    content = file.file.read()
    sep = "," if file.filename.endswith(".csv") else "\t"
    try:
        df = pd.read_csv(io.BytesIO(content), sep=sep, dtype=str, keep_default_na=False, encoding_errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV/TSV: {e}")
    stats = STORE.load_from_dataframe(
        df,
        text_col=text_col,
        context_col=context_col,
        id_col=id_col,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    INDEX_READY = True
    return {"ok": True, "stats": stats}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not INDEX_READY or STORE.mm.indices == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")
    # Retrieve final_k indices
    idxs = rerank_hybrid(STORE, req.query, top_k=req.top_k, final_k=req.final_k)
    if not idxs:
        return QueryResponse(answer="No relevant evidence found.", evidence=[], meta={"retrieved": 0})
    snippets = []
    ev = []
    for i in idxs:
        c = STORE.get_chunk(i)
        snippets.append(c.text)
        ev.append(EvidenceItem(text=c.text, hadm_id=c.hadm_id, context=c.context))
    # Generate
    draft = answer_with_context(req.query, snippets)
    refined = None
    if req.reflect:
        refined = reflect_and_refine(req.query, draft, snippets)
    return QueryResponse(answer=draft, refined_answer=refined, evidence=ev, meta={"retrieved": len(idxs)})

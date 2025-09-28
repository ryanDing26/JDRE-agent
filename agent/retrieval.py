# retrieval.py
import re
from typing import List, Dict, Tuple, Any, Set
import numpy as np
from functools import lru_cache

from rag_config import (
    USE_COSINE, TOP_K, FINAL_K, ALPHA_EMBED, BETA_KEYWORD, GAMMA_PHRASE,
    DELTA_OFFTOPIC, MMR_LAMBDA
)

# --- UMLS tagging imports ---
import spacy
import en_core_sci_md
from scispacy.abbreviation import AbbreviationDetector

_NLP = None

def _get_nlp(score_threshold: float = 0.85) -> spacy.language.Language:
    """
    Build (once) and return a SciSpaCy pipeline with abbreviation expansion + UMLS linker.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    nlp = en_core_sci_md.load()

    if "abbreviation_detector" not in nlp.pipe_names:
        try:
            nlp.add_pipe("abbreviation_detector")
        except Exception:
            nlp.add_pipe(AbbreviationDetector(nlp))

    # UMLS linker
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "resolve_abbreviations": True,
                "threshold": score_threshold,
            },
        )

    _NLP = nlp
    return _NLP

def umls_terms_only(text: str, top_k: int = 1, score_threshold: float = 0.85) -> List[str]:
    """
    Extract canonical UMLS concept names for entities in text.
    """
    nlp = _get_nlp(score_threshold)
    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")
    out = []
    for ent in doc.ents:
        cands = [(cui, score) for cui, score in (ent._.kb_ents or []) if score >= score_threshold]
        for cui, score in cands[:max(1, top_k)]:
            kb_ent = linker.kb.cui_to_entity.get(cui)
            if kb_ent:
                out.append(kb_ent.canonical_name.lower())
    return out


# --- Regex tokenizer ---
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

def simple_tokens(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


# --- Lightweight clinical synonym & domain heuristics ---
CLINICAL_SYNONYMS: Dict[str, List[str]] = {
    "red skin": ["erythema", "erythematous", "skin redness", "rash", "dermatitis", "exanthema", "exanthem", "flushing", "cutaneous"],
    "rash": ["eruption", "exanthem", "dermatitis", "skin rash"],
    "itching": ["pruritus", "pruritic", "itchy"],
    "shortness of breath": ["dyspnea", "sob", "breathlessness"],
    "chest pain": ["angina", "pressure", "tightness"],
    "fever": ["pyrexia", "febrile"],
    "swelling": ["edema"],
    "low oxygen": ["hypoxemia", "desaturation"],
    "high blood sugar": ["hyperglycemia"],
    "low blood sugar": ["hypoglycemia"],
}

DERM_TERMS: Set[str] = set("""
erythema erythematous rash dermatitis eczema urticaria hives cutaneous skin exanthem exanthema flushing
maculopapular plaques papules pruritus itchy erysipelas cellulitis erythematosus dermatology dermatologic
""".split())

HEMO_TERMS: Set[str] = set("""
blood hemoglobin hematocrit rbc wbc anemia anemic erythrocyte thrombocyte platelet coagulopathy hemolysis transfusion
""".split())


# --- Expansion logic (synonyms + UMLS) ---
@lru_cache(maxsize=2048)
def expand_query_terms(q: str) -> Tuple[str, ...]:
    tokens = simple_tokens(q)
    expanded = set([q.lower()])

    # 1. Handcrafted synonyms
    for t in tokens:
        if t in CLINICAL_SYNONYMS:
            expanded.update(s.lower() for s in CLINICAL_SYNONYMS[t])
    for phrase, syns in CLINICAL_SYNONYMS.items():
        if phrase in q.lower():
            expanded.update(s.lower() for s in syns)

    # 2. UMLS canonical names
    try:
        umls_names = umls_terms_only(q)
        expanded.update(umls_names)
    except Exception as e:
        print(f"[WARN] UMLS tagging failed: {e}")

    return tuple(sorted(expanded))


def combine_query_for_embedding(expanded_terms: Tuple[str, ...]) -> str:
    return " | ".join(expanded_terms)


def keyword_overlap_score(doc_text: str, expanded_terms: Tuple[str, ...]) -> float:
    dtoks = set(simple_tokens(doc_text))
    qtoks = set()
    for t in expanded_terms:
        qtoks.update(simple_tokens(t))
    if not qtoks:
        return 0.0
    overlap = len(dtoks.intersection(qtoks))
    return overlap / max(1, len(qtoks))


def exact_phrase_bonus(doc_text: str, q: str, expanded_terms: Tuple[str, ...]) -> float:
    low = doc_text.lower()
    bonus = 0.0
    if q.lower() in low:
        bonus += 1.0
    for t in expanded_terms:
        if len(t.split()) > 1 and t in low:
            bonus += 0.5
    return bonus


def offtopic_penalty(doc_text: str, q: str) -> float:
    qlow = q.lower()
    derm_hint = any(w in qlow for w in ["skin", "rash", "erythema", "dermatitis", "erythematous", "cutaneous"])
    if not derm_hint:
        return 0.0
    dtoks = set(simple_tokens(doc_text))
    if len(HEMO_TERMS.intersection(dtoks)) >= 2 and len(DERM_TERMS.intersection(dtoks)) == 0:
        return 1.0
    return 0.0


def mmr_select(
    candidates: List[Tuple[int, float, np.ndarray, str]],
    k: int,
    lambda_param: float
) -> List[int]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = [0]
    if len(candidates) == 1:
        return [candidates[0][0]]

    vecs = np.vstack([c[2] for c in candidates])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6
    nvecs = vecs / norms
    sim = nvecs @ nvecs.T

    while len(selected) < min(k, len(candidates)):
        best_idx = None
        best_score = -1e9
        for i in range(len(candidates)):
            if i in selected:
                continue
            rel = candidates[i][1]
            div = max(sim[i, j] for j in selected)
            mmr = lambda_param * rel - (1.0 - lambda_param) * div
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(best_idx)

    return [candidates[i][0] for i in selected]


def rerank_hybrid(
    store, query: str, top_k: int = TOP_K, final_k: int = FINAL_K
) -> List[int]:
    expanded = expand_query_terms(query)
    emb_query_str = combine_query_for_embedding(expanded)
    qvecs = store.mm.encode_query_multi(emb_query_str)

    agg = store.mm.search_multi(qvecs, top_k=top_k)

    cand_tuples: List[Tuple[int, float, np.ndarray, str]] = []
    for idx, embed_score in agg.items():
        doc = store.get_chunk(idx)
        kw_s = keyword_overlap_score(doc.text, expanded)
        phr = exact_phrase_bonus(doc.text, query, expanded)
        off = offtopic_penalty(doc.text, query)
        final = ALPHA_EMBED * embed_score               + BETA_KEYWORD * kw_s               + GAMMA_PHRASE * (1.0 if phr > 0 else 0.0)               - DELTA_OFFTOPIC * off
        pvec = store.primary_vec(idx)
        if pvec is None:
            continue
        cand_tuples.append((idx, final, pvec, doc.text))

    if not cand_tuples:
        return []

    selected_indices = mmr_select(cand_tuples, k=final_k, lambda_param=MMR_LAMBDA)
    return selected_indices


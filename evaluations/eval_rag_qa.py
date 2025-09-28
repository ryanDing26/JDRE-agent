#eval_rag_qa.py
# Ema's addition
"""
Evaluate Biomedical RAG on QA datasets (BIOSQ / PubMedQA / generic CSV/JSONL).

Usage examples:
  # evaluate PubMedQA.csv questions against a corpus.csv used to build the index
  python eval_rag_qa.py \
    --corpus corpus.csv --corpus_text_col TEXT --corpus_context_col CONTEXT --corpus_id_col DOC_ID \
    --qa pubmedqa.csv --qa_q_col question --qa_a_col answer --out results_pubmedqa.csv --reflect

Notes:
 - The script WILL build an in-memory index from --corpus using DocumentStore.load_from_dataframe.
 - The script calls your OpenAI-based `answer_with_context` (so ensure OPENAI_API_KEY configured).
 - Supports CSV/TSV/JSONL inputs for QA and corpus (auto-detect by extension).
"""

import os
import re
import csv
import json
import math
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Tuple
from agent.document_store import DocumentStore
from agent.retrieval import rerank_hybrid
from agent.generation import answer_with_context, reflect_and_refine


dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# ---------- Text normalization & SQuAD-like metrics ----------
def normalize_answer(s: str) -> str:
    """Lower, remove punctuation, articles and extra whitespace."""
    if s is None:
        return ""
    s = str(s).lower()
    # remove punctuation
    s = re.sub(r"[\.\,\!\?\:\;\-\_\(\)\[\]\"'/\\]", " ", s)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    if match == 0:
        return 0.0
    precision = match / len(pred_toks)
    recall = match / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

# ---------- dataset loaders ----------
def read_table(path: str) -> pd.DataFrame:
    """Read CSV/TSV/JSONL file into DataFrame; tries to be flexible."""
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="ignore")
    if path.endswith(".tsv") or path.endswith(".txt"):
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding_errors="ignore")
    # fallback: try pandas read_csv autodetect
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="ignore")

# ---------- main evaluator ----------
def evaluate(
    corpus_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    corpus_text_col: str = "TEXT",
    corpus_context_col: Optional[str] = "CONTEXT",
    corpus_id_col: Optional[str] = "DOC_ID",
    qa_q_col: str = "question",
    qa_a_col: str = "answer",
    top_k: int = None,
    final_k: int = None,
    reflect: bool = False,
    out_csv: str = "eval_results.csv",
    max_examples: Optional[int] = None
):
    # instantiate store and build index
    print("[eval] building index from corpus (this may take a while)")
    store = DocumentStore()
    stats = store.load_from_dataframe(
        corpus_df,
        text_col=corpus_text_col,
        context_col=corpus_context_col,
        id_col=corpus_id_col,
        # chunk_size and overlap default to rag_config values
    )
    print(f"[eval] index ready: {stats}")

    results = []
    total_em = 0.0
    total_f1 = 0.0
    n = 0
    retrieved_counts = []

    it = qa_df.iterrows()
    if max_examples:
        it = list(qa_df.iterrows())[:max_examples]

    for idx, row in tqdm(it, desc="QA items"):
        qid = row.get("id") or row.get("ID") or idx
        question = str(row.get(qa_q_col, "")).strip()
        gold = row.get(qa_a_col, "")
        # if gold is a list-like string (e.g. JSON array), coerce to first element
        if isinstance(gold, (list, tuple)):
            gold_answer = gold[0] if gold else ""
        else:
            # try parse json list
            try:
                parsed = json.loads(str(gold))
                if isinstance(parsed, (list, tuple)) and parsed:
                    gold_answer = parsed[0]
                else:
                    gold_answer = str(gold)
            except Exception:
                gold_answer = str(gold)

        # retrieval
        # call rerank_hybrid to get final indices; relies on globals in retrieval
        try:
            idxs = rerank_hybrid(store, question, top_k=top_k or None, final_k=final_k or None)
        except TypeError:
            # rerank_hybrid expects ints, allow None by replacing with default constants inside retrieval.py
            idxs = rerank_hybrid(store, question)
        if not idxs:
            pred = "No relevant evidence found."
            evidence_texts = []
            evidence_ids = []
        else:
            snippets = []
            evidence_texts = []
            evidence_ids = []
            for i in idxs:
                c = store.get_chunk(i)
                snippets.append(c.text)
                evidence_texts.append(c.text)
                evidence_ids.append(c.hadm_id)
            # generation
            try:
                draft = answer_with_context(question, snippets)
            except Exception as e:
                draft = f"[generation error] {e}"
            refined = None
            if reflect:
                try:
                    refined = reflect_and_refine(question, draft, snippets)
                except Exception as e:
                    refined = f"[reflect error] {e}"
            pred = refined if (reflect and refined) else draft

        em = exact_match(pred, gold_answer)
        f1 = f1_score(pred, gold_answer)

        total_em += em
        total_f1 += f1
        n += 1
        retrieved_counts.append(len(idxs) if idxs else 0)

        results.append({
            "id": qid,
            "question": question,
            "gold": gold_answer,
            "pred": pred,
            "em": em,
            "f1": f1,
            "retrieved_count": len(idxs) if idxs else 0,
            "evidence_ids": json.dumps(list(evidence_ids)),
            "evidence_texts": " ||| ".join(evidence_texts)[:10000],
        })

    # aggregate
    avg_em = total_em / max(1, n)
    avg_f1 = total_f1 / max(1, n)
    avg_ret = sum(retrieved_counts) / max(1, n)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    print(f"[eval] saved results to {out_csv}")

    summary = {
        "num_examples": n,
        "avg_em": avg_em,
        "avg_f1": avg_f1,
        "avg_retrieved": avg_ret,
    }
    print("[eval] SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary, df_out

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to corpus CSV/TSV/JSONL used to build index")
    parser.add_argument("--corpus_text_col", default="TEXT", help="column in corpus with textual content")
    parser.add_argument("--corpus_context_col", default="CONTEXT", help="optional column with context/metadata")
    parser.add_argument("--corpus_id_col", default="DOC_ID", help="optional unique id column for corpus rows")
    parser.add_argument("--qa", required=True, help="QA dataset file (CSV/TSV/JSONL)")
    parser.add_argument("--qa_q_col", default="question", help="QA dataset question column")
    parser.add_argument("--qa_a_col", default="answer", help="QA dataset answer column")
    parser.add_argument("--out", default="eval_results.csv", help="output CSV for predictions + metrics")
    parser.add_argument("--max_examples", type=int, default=None, help="limit number of QA examples for quick tests")
    parser.add_argument("--reflect", action="store_true", help="run reflect_and_refine after initial generation")
    parser.add_argument("--top_k", type=int, default=None, help="override top_k candidates from FAISS (optional)")
    parser.add_argument("--final_k", type=int, default=None, help="override final_k MMR selection (optional)")
    args = parser.parse_args()

    # read files
    print("[eval] loading corpus:", args.corpus)
    corpus_df = read_table(args.corpus)
    print("[eval] corpus rows:", len(corpus_df))
    print("[eval] loading qa:", args.qa)
    qa_df = read_table(args.qa)
    print("[eval] qa rows:", len(qa_df))

    summary, df_out = evaluate(
        corpus_df=corpus_df,
        qa_df=qa_df,
        corpus_text_col=args.corpus_text_col,
        corpus_context_col=(args.corpus_context_col or None),
        corpus_id_col=(args.corpus_id_col or None),
        qa_q_col=args.qa_q_col,
        qa_a_col=args.qa_a_col,
        top_k=args.top_k,
        final_k=args.final_k,
        reflect=args.reflect,
        out_csv=args.out,
        max_examples=args.max_examples,
    )
    summary.to_csv("qa_metrics.json")
    df_out.to_csv("qa_results.csv")

if __name__ == "__main__":
    main()
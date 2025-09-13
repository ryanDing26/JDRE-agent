# generation.py
from typing import List
from openai import OpenAI
from rag_config import OPENAI_API_KEY, DEFAULT_LLM_MODEL

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY or "REPLACE_WITH_YOUR_OPENAI_KEY" in OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not set in rag_config.py (OPENAI_API_KEY).")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def _build_context(snippets: List[str]) -> str:
    # small, efficient format; avoids long prompts
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))

def answer_with_context(question: str, snippets: List[str]) -> str:
    client = get_client()
    system = (
        "You are a careful biomedical assistant. "
        "Use ONLY the supplied evidence to answer. "
        "Cite snippets with [number]. If evidence is insufficient, say so."
    )
    user = f"Question: {question}\n\nEvidence:\n{_build_context(snippets)}\n\nAnswer:"
    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def reflect_and_refine(question: str, draft: str, snippets: List[str]) -> str:
    client = get_client()
    system = (
        "You are a strict biomedical verifier. Improve the draft only if needed and ensure all claims are supported by evidence."
    )
    user = (
        f"Question: {question}\n"
        f"Evidence:\n{_build_context(snippets)}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Instructions:\n"
        "- Verify support for each claim.\n"
        "- If something is missing, add it using the evidence.\n"
        "- Keep it concise and cite snippets like [1], [2].\n"
        "- If evidence is insufficient, say so.\n\n"
        "Refined answer:"
    )
    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1,
    )
    return resp.choices[0].message.content

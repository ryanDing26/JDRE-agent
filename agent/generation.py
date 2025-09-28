# generation.py
from typing import List
from openai import OpenAI
from rag_config import OPENAI_API_KEY, DEFAULT_LLM_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

def _build_context(snippets: List[str]) -> str:
    # small, efficient format; avoids long prompts
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))

def answer_with_context(question: str, snippets: List[str]) -> str:
    system = (
        "You are a careful biomedical assistant. "
        "Use ONLY the supplied evidence to answer. "
        "Cite snippets with [number]. If evidence is insufficient, say so."
    )
    user = f"Question: {question}\n\nEvidence:\n{_build_context(snippets)}\n\nAnswer:"
    resp = _client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def reflect_and_refine(question: str, draft: str, snippets: List[str]) -> str:
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
    resp = _client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1,
    )
    return resp.choices[0].message.content

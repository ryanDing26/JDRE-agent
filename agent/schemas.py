# schemas.py
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class LoadJSONRequest(BaseModel):
    rows: List[Dict[str, Any]]
    text_col: str = Field(default="TEXT")
    context_col: Optional[str] = Field(default="CONTEXT")
    id_col: Optional[str] = Field(default="HADM_ID")
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 12
    final_k: int = 5
    reflect: bool = False

class EvidenceItem(BaseModel):
    text: str
    hadm_id: Optional[Any] = None
    context: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    refined_answer: Optional[str] = None
    evidence: List[EvidenceItem]
    meta: Dict[str, Any] = {}

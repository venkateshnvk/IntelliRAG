from pydantic import BaseModel
from typing import Dict, Any


class QueryRequest(BaseModel):
    patient_id: str
    query: str


class QueryResponse(BaseModel):
    answer: Dict[str, Any]
    retrieval_confidence_score: float
    retrieval_confidence_level: str
    model_used: str | None
    latency_ms: float
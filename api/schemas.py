from pydantic import BaseModel
from typing import Dict, Any


# -------------------------------
# Existing Models (Keep as-is)
# -------------------------------

class QueryRequest(BaseModel):
    patient_id: str
    query: str


class QueryResponse(BaseModel):
    answer: Dict[str, Any]
    retrieval_confidence_score: float
    retrieval_confidence_level: str
    model_used: str | None
    latency_ms: float


# -------------------------------
# NEW: Evaluation Models
# -------------------------------

class EvaluationRequest(BaseModel):
    patient_id: str
    query: str
    reference_answer: str


class EvaluationResponse(BaseModel):
    answer: Dict[str, Any]
    ragas_metrics: Dict[str, float]

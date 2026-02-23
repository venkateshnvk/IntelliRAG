from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse
from llm.generator import RAGGenerator

router = APIRouter()
rag = RAGGenerator()

@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    try:
        result = rag.generate_answer(
            query=request.query,
            patient_id=request.patient_id
        )

        return QueryResponse(
            answer=result["answer"],
            retrieval_confidence_score=result["retrieval_confidence_score"],
            retrieval_confidence_level=result["retrieval_confidence_level"],
            model_used=result["model_used"],
            latency_ms=result["latency_ms"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# NEW: Get Unique Patient IDs
# ---------------------------------------------------------
@router.get("/patients")
def get_patients():

    try:
        results = rag.retriever.client.search(
            search_text="*",
            select=["patient_id"],
            top=1000
        )

        patient_ids = set()

        for r in results:
            patient_ids.add(r["patient_id"])

        return {
            "patients": sorted(list(patient_ids))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

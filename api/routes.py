from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse
from llm.generator import RAGGenerator
import time

router = APIRouter()
rag = RAGGenerator()


# ---------------------------------------------------------
# Query Classifier
# ---------------------------------------------------------

def classify_query(query: str):

    q = query.lower().strip()

    greetings = [
        "hi",
        "hello",
        "hey",
        "how are you",
        "good morning",
        "good evening",
        "who are you",
        "what can you do"
    ]

    if q in greetings:
        return "conversation"

    return "rag"


# ---------------------------------------------------------
# ASK ENDPOINT
# ---------------------------------------------------------

@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    try:

        query_type = classify_query(request.query)

        # -----------------------------------
        # Conversational Assistant
        # -----------------------------------

        if query_type == "conversation":

            start = time.time()

            response = rag.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are IntelliRAG, a helpful healthcare AI assistant."
                    },
                    {
                        "role": "user",
                        "content": request.query
                    }
                ],
                temperature=0.7
            )

            latency = (time.time() - start) * 1000

            answer = response.choices[0].message.content

            return QueryResponse(
                answer={
                    "answer": answer,
                    "evidence": [],
                    "source_report": None
                },
                retrieval_confidence_score=1.0,
                retrieval_confidence_level="High",
                model_used="gpt-4o-mini",
                latency_ms=latency
            )

        # -----------------------------------
        # RAG PIPELINE
        # -----------------------------------

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
# GET PATIENT IDS
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

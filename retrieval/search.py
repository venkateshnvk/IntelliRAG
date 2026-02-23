from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from embeddings.embedder import Embedder
from config.settings import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME
)


class Retriever:

    def __init__(self):
        self.embedder = Embedder()

        self.client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )

    # ---------------------------------------------------
    # Confidence Normalization (Human Friendly)
    # ---------------------------------------------------
    @staticmethod
    def _normalize_confidence(score: float) -> float:
        """
        Convert raw Azure vector score (~0.01–0.03)
        into 0–1 normalized scale.
        """

        MIN_SCORE = 0.01
        MAX_SCORE = 0.03

        # Clamp score within expected range
        score = max(MIN_SCORE, min(score, MAX_SCORE))

        normalized = (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)

        return round(normalized, 2)

    @staticmethod
    def _confidence_level(normalized_score: float) -> str:
        if normalized_score >= 0.7:
            return "High"
        elif normalized_score >= 0.4:
            return "Medium"
        else:
            return "Low"

    # ---------------------------------------------------
    # Hybrid Search
    # ---------------------------------------------------
    def hybrid_search(self, query: str, patient_id: str, top_k: int = 5):

        query_embedding = self.embedder.generate_embedding(query)

        results = self.client.search(
            search_text=query,
            vector_queries=[
                {
                    "kind": "vector",
                    "vector": query_embedding,
                    "fields": "embedding",
                    "k": top_k
                }
            ],
            filter=f"patient_id eq '{patient_id}'",
            top=top_k
        )

        documents = []

        for result in results:
            documents.append({
                "id": result["id"],
                "content": result["content"],
                "report_type": result["report_type"],
                "source_path": result["source_path"],
                "score": result.get("@search.score", 0.0)
            })

        if not documents:
            return {
                "documents": [],
                "raw_confidence": 0.0,
                "confidence_score": 0.0,
                "confidence_level": "Low"
            }

        # Use best score (more stable than average)
        best_score = max(doc["score"] for doc in documents)

        normalized_score = self._normalize_confidence(best_score)
        level = self._confidence_level(normalized_score)

        return {
            "documents": documents,
            "raw_confidence": round(best_score, 6),
            "confidence_score": normalized_score,
            "confidence_level": level
        }
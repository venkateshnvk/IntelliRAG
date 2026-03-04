from config.settings import (
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_FALLBACK_DEPLOYMENT
)


class ModelRouter:

    @staticmethod
    def choose_model(query: str, retrieval_confidence: float):

        query = query.lower().strip()

        # ---------------------------------
        # Simple fact lookup → Fast model
        # ---------------------------------
        simple_queries = [
            "name",
            "age",
            "gender",
            "blood type",
            "doctor",
            "diagnosis",
            "visit date"
        ]

        if any(word in query for word in simple_queries):
            return AZURE_OPENAI_CHAT_DEPLOYMENT


        # ---------------------------------
        # Short queries → Fast model
        # ---------------------------------
        if len(query.split()) <= 5:
            return AZURE_OPENAI_CHAT_DEPLOYMENT


        # ---------------------------------
        # Low retrieval confidence → Large model
        # ---------------------------------
        if retrieval_confidence < 0.25:
            return AZURE_OPENAI_FALLBACK_DEPLOYMENT


        # ---------------------------------
        # Complex reasoning
        # ---------------------------------
        complex_keywords = [
            "compare",
            "analyze",
            "trend",
            "risk",
            "why",
            "explain",
            "relationship",
            "impact",
            "cause"
        ]

        if any(word in query for word in complex_keywords):
            return AZURE_OPENAI_FALLBACK_DEPLOYMENT


        # ---------------------------------
        # Default → Fast model
        # ---------------------------------
        return AZURE_OPENAI_CHAT_DEPLOYMENT

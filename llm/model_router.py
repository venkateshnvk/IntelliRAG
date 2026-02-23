from config.settings import (
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_FALLBACK_DEPLOYMENT
)


class ModelRouter:

    @staticmethod
    def choose_model(query: str, retrieval_confidence: float):

        # Low retrieval confidence → escalate model
        if retrieval_confidence < 0.02:
            return AZURE_OPENAI_FALLBACK_DEPLOYMENT

        # Complex reasoning queries → escalate
        complex_keywords = [
            "compare",
            "analyze",
            "trend",
            "risk",
            "why",
            "explain",
            "relationship"
        ]

        if any(word in query.lower() for word in complex_keywords):
            return AZURE_OPENAI_FALLBACK_DEPLOYMENT

        return AZURE_OPENAI_CHAT_DEPLOYMENT
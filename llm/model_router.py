from config.settings import (
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_FALLBACK_DEPLOYMENT
)

class ModelRouter:

    @staticmethod
    def choose_model(query: str, retrieval_confidence: float):

        query = query.lower().strip()

        # ------------------------------------------------
        # 1️⃣ Short fact queries → fast model
        # ------------------------------------------------
        if len(query.split()) <= 5:
            return AZURE_OPENAI_CHAT_DEPLOYMENT


        # ------------------------------------------------
        # 2️⃣ Low retrieval confidence → bigger model
        # ------------------------------------------------
        if retrieval_confidence is not None and retrieval_confidence < 0.25:
            return AZURE_OPENAI_FALLBACK_DEPLOYMENT


        # ------------------------------------------------
        # 3️⃣ Complex reasoning queries
        # ------------------------------------------------
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


        # ------------------------------------------------
        # 4️⃣ Default → fast model
        # ------------------------------------------------
        return AZURE_OPENAI_CHAT_DEPLOYMENT

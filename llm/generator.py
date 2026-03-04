from openai import AzureOpenAI
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor

from retrieval.search import Retriever
from llm.model_router import ModelRouter

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION
)


class RAGGenerator:

    def __init__(self):

        self.retriever = Retriever()

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )

        self.intent_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    # -----------------------------------------------------
    # Intent Detection
    # -----------------------------------------------------

    def _detect_intent(self, query):

        q = query.lower().strip()

        if q in self.intent_cache:
            return self.intent_cache[q]

        prompt = f"""
Classify the query intent.

conversation → greetings or assistant questions
rag → patient record question

Examples:

hi → conversation
hello → conversation
how are you → conversation
who are you → conversation

patient name → rag
diagnosis → rag
medications → rag

Return only one word:

conversation
or
rag

Query: {query}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You classify user intent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        intent = response.choices[0].message.content.strip().lower()

        if "conversation" not in intent:
            intent = "rag"

        self.intent_cache[q] = intent

        return intent

    # -----------------------------------------------------
    # Direct Answer Extraction
    # -----------------------------------------------------

    def _direct_answer(self, query, docs):

        q = query.lower()

        patterns = {
            "patient name": r"Full Name:\s*([A-Za-z\s]+)",
            "age": r"Age:\s*(\d+)",
            "blood type": r"Blood Type:\s*([A-Za-z+-]+)",
            "diagnosis": r"Primary:\s*([A-Za-z\s]+)"
        }

        for key, pattern in patterns.items():

            if key in q:

                for doc in docs:

                    match = re.search(pattern, doc["content"], re.IGNORECASE)

                    if match:

                        value = match.group(1).strip()

                        return {
                            "answer": value,
                            "evidence": [match.group(0)],
                            "source_report": doc["report_type"]
                        }

        return None

    # -----------------------------------------------------
    # Conversation Handler
    # -----------------------------------------------------

    def _handle_conversation(self, query, start):

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are IntelliRAG, a healthcare AI assistant."
                },
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )

        latency = round((time.time() - start) * 1000, 2)

        return {
            "answer": {
                "answer": response.choices[0].message.content,
                "evidence": [],
                "source_report": None
            },
            "retrieval_confidence_score": 1.0,
            "retrieval_confidence_level": "High",
            "model_used": "gpt-4o-mini",
            "latency_ms": latency
        }

    # -----------------------------------------------------
    # Context Builder
    # -----------------------------------------------------

    def _build_context(self, docs):

        grouped = {}

        for doc in docs:
            grouped.setdefault(doc["report_type"], []).append(doc["content"])

        sections = []

        for rtype, contents in grouped.items():

            section = f"\n### {rtype.upper()} REPORTS\n"

            for c in contents:
                section += c + "\n"

            sections.append(section)

        return "\n".join(sections)

    # -----------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------

    def generate_answer(self, query, patient_id, top_k=3):

        start = time.time()

        intent_future = self.executor.submit(self._detect_intent, query)

        retrieval_future = self.executor.submit(
            self.retriever.hybrid_search,
            query,
            patient_id,
            top_k
        )

        intent = intent_future.result()
        retrieval = retrieval_future.result()

        if intent == "conversation":
            return self._handle_conversation(query, start)

        docs = retrieval["documents"]
        score = retrieval["confidence_score"]
        level = retrieval["confidence_level"]

        if not docs:

            latency = round((time.time() - start) * 1000, 2)

            return {
                "answer": {
                    "answer": "I could not find that information in the patient's records.",
                    "evidence": [],
                    "source_report": None
                },
                "retrieval_confidence_score": score,
                "retrieval_confidence_level": level,
                "model_used": None,
                "latency_ms": latency
            }

        fast = self._direct_answer(query, docs)

        if fast:

            latency = round((time.time() - start) * 1000, 2)

            return {
                "answer": fast,
                "retrieval_confidence_score": 1.0,
                "retrieval_confidence_level": "High",
                "model_used": "direct-extraction",
                "latency_ms": latency
            }

        docs = docs[:3]

        context = self._build_context(docs)

        model = ModelRouter.choose_model(query, score)

        system_prompt = """
You are IntelliRAG.

Answer questions using patient medical records.

Rules:
Use only the provided records.
If answer not found say:
"I could not find that information in the patient's records."

Return JSON:

{
 "answer": "text",
 "evidence": ["snippet"],
 "source_report": "medical"
}
"""

        user_prompt = f"""
Patient ID: {patient_id}

Records:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content

        try:
            parsed = json.loads(raw)
        except:
            parsed = {
                "answer": "Unexpected response format.",
                "evidence": [],
                "source_report": None
            }

        latency = round((time.time() - start) * 1000, 2)

        return {
            "answer": parsed,
            "retrieval_confidence_score": score,
            "retrieval_confidence_level": level,
            "model_used": model,
            "latency_ms": latency
        }

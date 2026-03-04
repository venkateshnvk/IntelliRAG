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

        # Intent cache for repeated queries
        self.intent_cache = {}

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=2)

    # -----------------------------------------------------
    # Intent Detection with Cache
    # -----------------------------------------------------
    def _detect_intent(self, query):

        query_key = query.lower().strip()

        if query_key in self.intent_cache:
            return self.intent_cache[query_key]

        prompt = f"""
Classify the user query into one of two intents:

conversation
rag

Examples:
hi → conversation
hello → conversation
how are you → conversation
who are you → conversation

patient name → rag
what is the diagnosis → rag
what medication is prescribed → rag

Return only one word.

Query: {query}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intent classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        intent = response.choices[0].message.content.strip().lower()

        if "conversation" not in intent:
            intent = "rag"

        self.intent_cache[query_key] = intent

        return intent

    # -----------------------------------------------------
    # Direct Answer Extraction
    # -----------------------------------------------------
    def _direct_answer(self, query, docs):

        query_lower = query.lower()

        patterns = {
            "patient name": r"Full Name:\s*([A-Za-z\s]+)",
            "age": r"Age:\s*(\d+)",
            "blood type": r"Blood Type:\s*([A-Za-z+-]+)",
            "primary diagnosis": r"Primary:\s*([A-Za-z\s]+)",
            "diagnosis": r"Primary:\s*([A-Za-z\s]+)"
        }

        for key, pattern in patterns.items():

            if key in query_lower:

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
    def _handle_conversation(self, query, start_time):

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are IntelliRAG, an enterprise healthcare AI assistant.

You help users understand patient medical records such as:
- diagnoses
- medications
- scan reports
- billing
- insurance

Respond in a friendly and professional conversational tone.
"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "answer": {
                "answer": answer,
                "evidence": [],
                "source_report": None
            },
            "retrieval_confidence_score": 1.0,
            "retrieval_confidence_level": "High",
            "model_used": "gpt-4o-mini",
            "latency_ms": latency_ms
        }

    # -----------------------------------------------------
    # Build Structured Context
    # -----------------------------------------------------
    def _build_structured_context(self, retrieved_docs):

        grouped = {}

        for doc in retrieved_docs:
            report_type = doc["report_type"]
            grouped.setdefault(report_type, []).append(doc["content"])

        context_sections = []

        for report_type, contents in grouped.items():

            section = f"\n### {report_type.upper()} REPORTS\n"

            for content in contents:
                section += content + "\n"

            context_sections.append(section)

        return "\n".join(context_sections)

    # -----------------------------------------------------
    # Main RAG Pipeline
    # -----------------------------------------------------
    def generate_answer(self, query, patient_id, top_k=3):

        start_time = time.time()

        # -------------------------------------------------
        # Parallel Intent Detection + Retrieval
        # -------------------------------------------------

        intent_future = self.executor.submit(self._detect_intent, query)

        retrieval_future = self.executor.submit(
            self.retriever.hybrid_search,
            query,
            patient_id,
            top_k
        )

        intent = intent_future.result()
        retrieval_output = retrieval_future.result()

        print("\nDetected intent:", intent)

        if intent == "conversation":
            return self._handle_conversation(query, start_time)

        # -------------------------------------------------
        # Retrieval Results
        # -------------------------------------------------

        retrieved_docs = retrieval_output["documents"]
        confidence_score = retrieval_output["confidence_score"]
        confidence_level = retrieval_output["confidence_level"]

        if not retrieved_docs:

            latency_ms = round((time.time() - start_time) * 1000, 2)

            return {
                "answer": {
                    "answer": "I could not find that information in the patient's records.",
                    "evidence": [],
                    "source_report": None
                },
                "retrieval_confidence_score": confidence_score,
                "retrieval_confidence_level": confidence_level,
                "model_used": None,
                "latency_ms": latency_ms
            }

        # -------------------------------------------------
        # Direct Answer Extraction (fast path)
        # -------------------------------------------------

        direct_answer = self._direct_answer(query, retrieved_docs)

        if direct_answer:

            latency_ms = round((time.time() - start_time) * 1000, 2)

            return {
                "answer": direct_answer,
                "retrieval_confidence_score": 1.0,
                "retrieval_confidence_level": "High",
                "model_used": "direct-extraction",
                "latency_ms": latency_ms
            }

        # Reduce context size
        retrieved_docs = retrieved_docs[:3]

        structured_context = self._build_structured_context(retrieved_docs)

        # -------------------------------------------------
        # Model Router
        # -------------------------------------------------

        model_to_use = ModelRouter.choose_model(query, confidence_score)

        # -------------------------------------------------
        # LLM Prompt
        # -------------------------------------------------

        system_prompt = """
You are IntelliRAG, an enterprise healthcare AI assistant.

Your job is to answer questions using patient medical records.

Guidelines:
- Respond in clear, natural, professional English.
- Your answer should sound like a human healthcare assistant.
- Always base your answer strictly on the provided patient records.
- Do not invent information.

If the answer is not found in the records, respond:
"I could not find that information in the patient's records."

Return JSON in this format:

{
  "answer": "natural language explanation",
  "evidence": ["exact snippet from records"],
  "source_report": "medical | billing | insurance"
}
"""

        user_prompt = f"""
Patient ID: {patient_id}

Patient Records:
{structured_context}

User Question:
{query}

Provide a clear answer using the records.
"""

        # -------------------------------------------------
        # LLM Call
        # -------------------------------------------------

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        answer_raw = response.choices[0].message.content

        try:
            parsed_answer = json.loads(answer_raw)

        except json.JSONDecodeError:

            parsed_answer = {
                "answer": "The system generated an unexpected response format.",
                "evidence": [],
                "source_report": None
            }

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "answer": parsed_answer,
            "retrieval_confidence_score": confidence_score,
            "retrieval_confidence_level": confidence_level,
            "model_used": model_to_use,
            "latency_ms": latency_ms
        }

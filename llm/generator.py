from openai import AzureOpenAI
import json
import time

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

        self.greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening"
        ]

    # -----------------------------------------------------
    # Greeting Detection
    # -----------------------------------------------------
    def _is_greeting(self, query):

        query = query.lower().strip()

        return any(g in query for g in self.greetings)

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
    # Main RAG Execution
    # -----------------------------------------------------
    def generate_answer(self, query, patient_id, top_k=3):

        start_time = time.time()

        # -------------------------------------------------
        # Step 0 — Greeting Handling
        # -------------------------------------------------
        if self._is_greeting(query):

            latency_ms = round((time.time() - start_time) * 1000, 2)

            return {
                "answer": {
                    "answer": "Hello! I'm IntelliRAG, your healthcare assistant. You can ask about the patient's diagnosis, medications, medical reports, billing details, or insurance information.",
                    "evidence": [],
                    "source_report": None
                },
                "retrieval_confidence_score": None,
                "retrieval_confidence_level": None,
                "model_used": None,
                "latency_ms": latency_ms
            }

        # -------------------------------------------------
        # Step 1 — Retrieval
        # -------------------------------------------------
        retrieval_output = self.retriever.hybrid_search(
            query=query,
            patient_id=patient_id,
            top_k=top_k
        )

        retrieved_docs = retrieval_output["documents"]
        confidence_score = retrieval_output["confidence_score"]
        confidence_level = retrieval_output["confidence_level"]

        # -------------------------------------------------
        # If nothing retrieved
        # -------------------------------------------------
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
        # Reduce context (faster LLM)
        # -------------------------------------------------
        retrieved_docs = retrieved_docs[:3]

        structured_context = self._build_structured_context(retrieved_docs)

        # -------------------------------------------------
        # Step 2 — Model Routing
        # -------------------------------------------------
        model_to_use = ModelRouter.choose_model(query, confidence_score)

        print("\n--- RAG DEBUG INFO ---")
        print("Query:", query)
        print("Patient ID:", patient_id)
        print("Retrieval confidence:", confidence_score)
        print("Confidence level:", confidence_level)
        print("Model selected:", model_to_use)
        print("----------------------\n")

        # -------------------------------------------------
        # Step 3 — Prompt
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
        # Step 4 — LLM Call
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

        # -------------------------------------------------
        # Step 5 — Safe JSON Parsing
        # -------------------------------------------------
        try:
            parsed_answer = json.loads(answer_raw)

        except json.JSONDecodeError:

            parsed_answer = {
                "answer": "The system generated an unexpected response format.",
                "evidence": [],
                "source_report": None
            }

        # -------------------------------------------------
        # Step 6 — Latency
        # -------------------------------------------------
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # -------------------------------------------------
        # Final Response
        # -------------------------------------------------
        return {
            "answer": parsed_answer,
            "retrieval_confidence_score": confidence_score,
            "retrieval_confidence_level": confidence_level,
            "model_used": model_to_use,
            "latency_ms": latency_ms
        }

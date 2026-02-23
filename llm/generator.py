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

    # -----------------------------------------------------
    # Build grouped structured context
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
    def generate_answer(self, query, patient_id, top_k=5):

        start_time = time.time()  # ⏱ Total pipeline start

        # ---------------------------
        # Step 1 — Retrieval
        # ---------------------------
        retrieval_output = self.retriever.hybrid_search(
            query=query,
            patient_id=patient_id,
            top_k=top_k
        )

        retrieved_docs = retrieval_output["documents"]
        confidence_score = retrieval_output["confidence_score"]
        confidence_level = retrieval_output["confidence_level"]

        # Handle empty retrieval safely
        if not retrieved_docs:
            latency_ms = round((time.time() - start_time) * 1000, 2)

            return {
                "answer": {
                    "message": "Information not available in patient records."
                },
                "retrieval_confidence_score": confidence_score,
                "retrieval_confidence_level": confidence_level,
                "model_used": None,
                "latency_ms": latency_ms
            }

        structured_context = self._build_structured_context(retrieved_docs)

        # ---------------------------
        # Step 2 — Model Routing
        # ---------------------------
        model_to_use = ModelRouter.choose_model(query, confidence_score)

        print("Model selected:", model_to_use)
        print("Retrieval confidence score:", confidence_score)
        print("Retrieval confidence level:", confidence_level)

        # ---------------------------
        # Step 3 — Strict JSON Prompt
        # ---------------------------
        system_prompt = """
You are a healthcare clinical assistant.

STRICT RULES:
- Use ONLY information present in the patient records.
- Return structured JSON.
- For each field include:
    - value
    - supporting_evidence copied exactly from records.
- Do NOT add reasoning.
- Do NOT summarize.
- Do NOT output text outside JSON.
"""

        user_prompt = f"""
Patient ID: {patient_id}

Patient Records:
{structured_context}

Question:
{query}

Return JSON format only.
"""

        # ---------------------------
        # Step 4 — LLM Call
        # ---------------------------
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

        # ---------------------------
        # Step 5 — Safe JSON Parsing
        # ---------------------------
        try:
            parsed_answer = json.loads(answer_raw)
        except json.JSONDecodeError:
            parsed_answer = {
                "error": "Model returned invalid JSON",
                "raw_output": answer_raw
            }

        # ---------------------------
        # Step 6 — Compute Latency
        # ---------------------------
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # ---------------------------
        # Final Structured Response
        # ---------------------------
        return {
            "answer": parsed_answer,
            "retrieval_confidence_score": confidence_score,
            "retrieval_confidence_level": confidence_level,
            "model_used": model_to_use,
            "latency_ms": latency_ms
        }
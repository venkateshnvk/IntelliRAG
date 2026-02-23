from evaluation.ragas_eval import RAGASEvaluator

# -------------------------------------------------
# Test Configuration
# -------------------------------------------------

query = "What is the primary diagnosis of the patient?"
patient_id = "PAT1025"

# 🔴 IMPORTANT:
# This must be the TRUE expected answer from records
reference_answer = "Hypertension"


# -------------------------------------------------
# Run Evaluation
# -------------------------------------------------

print("\n==============================")
print("IntelliRAG - RAGAS Evaluation")
print("==============================")
print(f"Patient ID: {patient_id}")
print(f"Query: {query}")
print("------------------------------")

evaluator = RAGASEvaluator()

result = evaluator.evaluate_single_query(
    query=query,
    patient_id=patient_id,
    reference_answer=reference_answer
)

# -------------------------------------------------
# Output Results
# -------------------------------------------------

print("\nGenerated Answer:")
print(result["answer"])

print("\nLatency (seconds):")
print(result["latency_seconds"])

print("\nRAGAS Scores:")
print(result["ragas_scores"])
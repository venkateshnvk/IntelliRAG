from retrieval.search import Retriever

retriever = Retriever()

query = "What is the primary diagnosis of the patient?"

results = retriever.hybrid_search(
    query=query,
    patient_id="PAT1000",
    top_k=5
)

for i, r in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Score:", r.get("score"))   # ✅ FIXED
    print("Report Type:", r.get("report_type"))
    print("Source:", r.get("source_path"))
    print("Content:\n", r.get("content"))
from llm.generator import RAGGenerator


def main():

    patient_id = "PAT1023"
    query = "What is the primary diagnosis of the patient?"

    print("\n==============================")
    print("IntelliRAG - Test RAG")
    print("==============================")
    print(f"Patient ID: {patient_id}")
    print(f"Query: {query}")
    print("------------------------------")

    try:
        rag = RAGGenerator()

        answer = rag.generate_answer(
            query=query,
            patient_id=patient_id,
            top_k=5
        )

        print("\n✅ Final Answer:\n")
        print(answer)

    except Exception as e:
        print("\n❌ ERROR OCCURRED:")
        print(str(e))


if __name__ == "__main__":
    main()
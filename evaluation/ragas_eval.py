from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from llm.generator import RAGGenerator
from retrieval.search import Retriever

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
)

import time


class RAGASEvaluator:

    def __init__(self):

        self.generator = RAGGenerator()
        self.retriever = Retriever()

        azure_llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0
        )

        self.evaluator_llm = LangchainLLMWrapper(azure_llm)

        azure_embeddings = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )

        self.evaluator_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)

    def evaluate_single_query(self, query, patient_id, reference_answer):

        # ----------------------------
        # Measure Latency
        # ----------------------------
        start_time = time.time()

        retrieval_output = self.retriever.hybrid_search(
            query=query,
            patient_id=patient_id,
            top_k=5
        )

        retrieved_docs = retrieval_output["documents"]
        contexts = [doc["content"] for doc in retrieved_docs]

        rag_output = self.generator.generate_answer(
            query=query,
            patient_id=patient_id
        )

        end_time = time.time()
        latency = round(end_time - start_time, 3)

        answer_dict = rag_output["answer"]

        if isinstance(answer_dict, dict):
            answer_text = " ".join(str(v) for v in answer_dict.values())
        else:
            answer_text = str(answer_dict)

        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer_text],
            "contexts": [contexts],
            "reference": [reference_answer]
        })

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_correctness,
                context_precision,
                context_recall
            ],
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings
        )

        return {
            "ragas_scores": result,
            "latency_seconds"   : latency,
            "answer": answer_text
        }
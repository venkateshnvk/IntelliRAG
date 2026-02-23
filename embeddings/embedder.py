from openai import AzureOpenAI
from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
)


class Embedder:

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def generate_embedding(self, text: str):
        response = self.client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=text
        )
        return response.data[0].embedding
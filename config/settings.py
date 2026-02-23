import os
from dotenv import load_dotenv

load_dotenv()

# ============================
# Azure AI Search
# ============================

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# ============================
# Azure OpenAI
# ============================

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Embedding deployment (text-embedding-3-large)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Chat deployment (gpt-4o-mini)
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_FALLBACK_DEPLOYMENT = os.getenv("AZURE_OPENAI_FALLBACK_DEPLOYMENT")

# ============================
# Azure Blob Storage
# ============================

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")


# ============================
# Safety Checks (Optional but Recommended)
# ============================

required_vars = [
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER,
]

if not all(required_vars):
    raise ValueError("❌ One or more environment variables are missing. Check your .env file.")
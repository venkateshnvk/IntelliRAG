from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from config.settings import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME
)


class Indexer:

    def __init__(self):
        self.client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )

    # =========================
    # Upload
    # =========================
    def upload_documents(self, documents):
        result = self.client.upload_documents(documents=documents)
        print(f"Uploaded {len(documents)} documents.")
        return result

    # =========================
    # Delete by file_hash
    # =========================
    def delete_by_file_hash(self, file_hash):
        print(f"Deleting documents with file_hash: {file_hash}")

        results = self.client.search(
            search_text="*",
            filter=f"file_hash eq '{file_hash}'",
            select=["id"],
            top=1000
        )

        ids = [{"id": r["id"]} for r in results]

        if ids:
            self.client.delete_documents(documents=ids)
            print(f"Deleted {len(ids)} documents for file_hash.")
        else:
            print("No matching documents found for file_hash.")

    # =========================
    # Delete by patient_id
    # =========================
    def delete_by_patient_id(self, patient_id):
        print(f"Deleting documents for patient_id: {patient_id}")

        results = self.client.search(
            search_text="*",
            filter=f"patient_id eq '{patient_id}'",
            select=["id"],
            top=1000
        )

        ids = [{"id": r["id"]} for r in results]

        if ids:
            self.client.delete_documents(documents=ids)
            print(f"Deleted {len(ids)} documents for patient.")
        else:
            print("No documents found for patient.")
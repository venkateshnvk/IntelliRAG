from ingestion.chunker import Chunker
from embeddings.embedder import Embedder
from vectorstore.indexer import Indexer
from ingestion.parser import Parser
from ingestion.metadata_builder import MetadataBuilder
from azure.storage.blob import BlobServiceClient
from config.settings import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER
)

import re


class IngestionPipeline:

    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.indexer = Indexer()
        self.parser = Parser()

    # -------------------------------------------
    # Extract Full Name from ALL document types
    # -------------------------------------------
    def _extract_full_name(self, text: str) -> str:
        """
        Extract patient full name from medical, insurance,
        medication, billing, and scanning JSON documents.
        """

        patterns = [
            # Medical PDF
            r"Full Name:\s*\n?([A-Za-z\s]+)",

            # Billing
            r"Patient Name:\s*\n?([A-Za-z\s]+)",

            # Insurance HTML
            r"Policy Holder Name:\s*\n?([A-Za-z\s]+)",

            # Medication DOCX
            r"Patient:\s*([A-Za-z\s]+)\s*\(",

            # JSON simple format
            r"patient_name:\s*([A-Za-z\s]+)",

            # JSON quoted format
            r'"patient_name"\s*:\s*"([A-Za-z\s]+)"'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "Unknown"

    # -------------------------------------------
    # Main Blob Processing
    # -------------------------------------------
    def process_blob(self, blob_name: str):

        print(f"\n📂 Processing blob: {blob_name}")
        print("Using container:", AZURE_STORAGE_CONTAINER)

        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )

        container_client = blob_service_client.get_container_client(
            AZURE_STORAGE_CONTAINER
        )

        blob_client = container_client.get_blob_client(blob_name)

        # Download file
        file_bytes = blob_client.download_blob().readall()

        # Parse content
        text = self.parser.parse(blob_name, file_bytes)

        if not text or not text.strip():
            print("⚠ Empty document. Skipping:", blob_name)
            return

        # Extract patient_id from filename
        filename = blob_name.split("/")[-1]
        patient_id = filename.split("_")[0]

        # Extract full_name
        full_name = self._extract_full_name(text)

        # Determine report type
        folder_name = blob_name.split("/")[0].lower()

        if "medical" in folder_name:
            report_type = "medical"
        elif "scanning" in folder_name:
            report_type = "scan"
        elif "medication" in folder_name:
            report_type = "medication"
        elif "billing" in folder_name:
            report_type = "billing"
        elif "insurance" in folder_name:
            report_type = "insurance"
        else:
            report_type = "unknown"

        # Chunk document
        chunks = self.chunker.split_text(text)

        if not chunks:
            print("⚠ No chunks generated. Skipping:", blob_name)
            return

        # Build metadata
        documents = MetadataBuilder.build_metadata(
            patient_id=patient_id,
            full_name=full_name,
            report_type=report_type,
            source_path=blob_name,
            file_bytes=file_bytes,
            chunks=chunks,
            document_date=None
        )

        # Deduplicate by file_hash
        file_hash = documents[0]["file_hash"]
        self.indexer.delete_by_file_hash(file_hash)

        # Generate embeddings
        for doc in documents:
            doc["embedding"] = self.embedder.generate_embedding(doc["content"])

        # -------------------------------------------
        # SAFE BATCH UPLOAD (Prevents Azure abort)
        # -------------------------------------------
        batch_size = 50
        total_uploaded = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.indexer.upload_documents(batch)
            total_uploaded += len(batch)

        print(f"✅ Uploaded {total_uploaded} chunks for {patient_id} ({full_name})")
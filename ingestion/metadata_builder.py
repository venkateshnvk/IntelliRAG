import hashlib
from datetime import datetime
import os


class MetadataBuilder:

    @staticmethod
    def generate_file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def build_metadata(
        patient_id: str,
        full_name: str,      # ✅ NEW PARAM
        report_type: str,
        source_path: str,
        file_bytes: bytes,
        chunks: list,
        document_date: str = None
    ):
        """
        Builds enriched metadata for each chunk.
        """

        file_name = os.path.basename(source_path)
        file_hash = MetadataBuilder.generate_file_hash(file_bytes)
        ingested_at = datetime.utcnow().isoformat() + "Z"

        documents = []
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):

            doc = {
                "id": f"{patient_id}_{report_type}_{file_hash[:8]}_{idx}",
                "patient_id": patient_id,
                "full_name": full_name,   # ✅ STORED CLEANLY
                "report_type": report_type,
                "source_path": source_path,
                "file_name": file_name,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "document_date": document_date,
                "ingested_at": ingested_at,
                "file_hash": file_hash,
                "content": chunk
            }

            documents.append(doc)

        return documents
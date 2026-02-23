from ingestion.pipeline import IngestionPipeline
from azure.storage.blob import BlobServiceClient
from config.settings import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER
)


def reindex_all():

    print("🔁 Starting full re-index...")

    pipeline = IngestionPipeline()

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(
        AZURE_STORAGE_CONTAINER
    )

    blobs = container_client.list_blobs()

    count = 0

    for blob in blobs:
        print(f"Processing: {blob.name}")
        pipeline.process_blob(blob.name)
        count += 1

    print(f"\n✅ Re-index completed. Total files processed: {count}")


if __name__ == "__main__":
    reindex_all()
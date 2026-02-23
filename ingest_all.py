from azure.storage.blob import BlobServiceClient
from ingestion.pipeline import IngestionPipeline
from config.settings import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER
)

pipeline = IngestionPipeline()

blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

container_client = blob_service_client.get_container_client(
    AZURE_STORAGE_CONTAINER
)

blobs = container_client.list_blobs()

for blob in blobs:
    pipeline.process_blob(blob.name)

print("✅ All blobs processed.")
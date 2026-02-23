# debug_billing.py
from azure.storage.blob import BlobServiceClient
from config.settings import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER

client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

container = client.get_container_client(AZURE_STORAGE_CONTAINER)

for blob in container.list_blobs(name_starts_with="04_patient_billings"):
    print(blob.name)
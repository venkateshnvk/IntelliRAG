import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential

from config.settings import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME
)


def create_index():

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)

    client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=credential
    )

    fields = [

        # -----------------------
        # Primary key
        # -----------------------
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True
        ),

        # -----------------------
        # Core metadata
        # -----------------------
        SimpleField(
            name="patient_id",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        # ✅ NEW FIELD
        SimpleField(
            name="full_name",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="report_type",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="source_path",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="file_name",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="file_hash",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),

        SimpleField(
            name="total_chunks",
            type=SearchFieldDataType.Int32
        ),

        SimpleField(
            name="document_date",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="ingested_at",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True
        ),

        # -----------------------
        # Searchable content
        # -----------------------
        SearchableField(
            name="content",
            type=SearchFieldDataType.String
        ),

        # -----------------------
        # Vector field
        # -----------------------
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="vector-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="vector-algorithm"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="vector-algorithm"
            )
        ]
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    client.create_or_update_index(index)

    print("✅ Azure AI Search index created successfully.")


if __name__ == "__main__":
    create_index()
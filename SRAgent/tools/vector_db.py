# import
## batteries
import os
import sys

## 3rd party
import chromadb
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_chroma import Chroma


# functions
def verify_collection(
    persistent_client: chromadb.PersistentClient, collection_name: str
) -> None:
    """
    Verify that the collection exists and has documents
    Args:
        persistent_client: The persistent Chroma client
        collection_name: The name of the collection to verify
    Returns:
        None
    Raises:
        Exception: If the collection does not exist or has no documents
    """
    try:
        collection = persistent_client.get_collection(collection_name)
        count = collection.count()
        print(
            f"Found {count} documents in collection '{collection_name}'",
            file=sys.stdout,
        )
    except Exception as e:
        msg = f"Error accessing collection: {e}"
        msg += f"\nAvailable collections: {persistent_client.list_collections()}"
        raise Exception(msg)


def load_vector_store(chroma_path: str, collection_name: str = "uberon") -> Chroma:
    """
    Load a Chroma vector store from the specified path.
    Args:
        chroma_path: The path to the Chroma DB directory
        collection_name: The name of the collection to load
    Returns:
        A Chroma vector store
    Raises:
        FileNotFoundError: If the Chroma DB directory does not exist
        Exception: If the collection does not exist or has no documents
    """
    # Ensure the path exists
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"Chroma DB directory not found: {chroma_path}")

    # Initialize embeddings. Azure OpenAI requires a dedicated embedding deployment.
    embedding_provider = os.getenv("SRAGENT_EMBEDDING_PROVIDER")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if embedding_provider is None and os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
        embedding_provider = "azure_openai"
    elif embedding_provider is None:
        embedding_provider = "openai"

    embedding_provider = embedding_provider.lower()

    if embedding_provider == "azure_openai":
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        azure_api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or "2024-12-01-preview"
        )

        if not azure_endpoint or not azure_api_key or not azure_deployment:
            raise ValueError(
                "Azure OpenAI embeddings require AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            )

        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", embedding_model),
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
    else:
        if os.getenv("OPENAI_API_KEY") is None and os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI embeddings were selected, but OPENAI_API_KEY is not set. "
                "Set OPENAI_API_KEY or configure Azure embeddings with "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
            )
        embeddings = OpenAIEmbeddings(model=embedding_model)

    # Load the persistent Chroma client
    persistent_client = chromadb.PersistentClient(path=chroma_path)

    # Load the existing vector store
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vector_store

"""
Vector store integrations for django-chain.
"""

from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def get_vector_store(store_type: str, embedding_function: Embeddings, **kwargs) -> VectorStore:
    """
    Get a vector store instance for the specified type.

    Args:
        store_type: The vector store type (e.g., 'pgvector', 'chroma')
        embedding_function: The embedding function to use
        **kwargs: Additional arguments for the vector store

    Returns:
        A configured vector store instance

    Raises:
        ImportError: If the required store package is not installed
        ValueError: If the store type is not supported
    """
    if store_type == "pgvector":
        from .pgvector import get_pgvector_store

        return get_pgvector_store(embedding_function=embedding_function, **kwargs)
    elif store_type == "chroma":
        from .chroma import get_chroma_store

        return get_chroma_store(embedding_function=embedding_function, **kwargs)
    elif store_type == "pinecone":
        from .pinecone import get_pinecone_store

        return get_pinecone_store(embedding_function=embedding_function, **kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")


def add_documents(
    store: VectorStore,
    texts: list[str],
    metadatas: Optional[list[dict[str, Any]]] = None,
) -> None:
    """
    Add documents to a vector store.

    Args:
        store: The vector store instance
        texts: List of text documents to add
        metadatas: Optional list of metadata dictionaries for each document
    """
    store.add_texts(texts=texts, metadatas=metadatas)


def retrieve_documents(store: VectorStore, query: str, k: int = 4) -> list[dict[str, Any]]:
    """
    Retrieve documents from a vector store.

    Args:
        store: The vector store instance
        query: The search query
        k: Number of documents to retrieve

    Returns:
        List of retrieved documents with their metadata
    """
    return store.similarity_search(query=query, k=k)

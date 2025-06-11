"""
PGVector integration for django-chain.
"""

from django.db import connections
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.embeddings import BaseEmbeddings
from langchain_core.vectorstores import VectorStore


def get_pgvector_store(
    embedding_function: BaseEmbeddings,
    collection_name: str = "langchain_documents",
    db_alias: str = "default",
    **kwargs,
) -> VectorStore:
    """
    Get a PGVector store instance.

    Args:
        embedding_function: The embedding function to use
        collection_name: Name of the collection to use
        db_alias: Django database alias to use
        **kwargs: Additional arguments for PGVector

    Returns:
        A configured PGVector instance

    Raises:
        ValueError: If the database engine is not PostgreSQL
    """
    db_settings = connections[db_alias].settings_dict

    # Verify we're using PostgreSQL
    if not db_settings.get("ENGINE", "").endswith("postgresql"):
        raise ValueError("PGVector requires a PostgreSQL database")

    # Construct connection string
    connection_string = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=db_settings.get("HOST", "localhost"),
        port=db_settings.get("PORT", "5432"),
        database=db_settings.get("NAME"),
        user=db_settings.get("USER"),
        password=db_settings.get("PASSWORD"),
    )

    return PGVector(
        connection_string=connection_string,
        embedding_function=embedding_function,
        collection_name=collection_name,
        **kwargs,
    )

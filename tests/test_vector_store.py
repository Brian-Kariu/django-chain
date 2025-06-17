"""
Tests for vector store manager.
"""

from unittest.mock import MagicMock, patch

import pytest
from django.conf import settings
from django.test import TestCase

from django_chain.exceptions import VectorStoreError
from django_chain.services.vector_store_manager import VectorStoreManager


@pytest.mark.skip()
class TestVectorStoreManager(TestCase):
    """Test cases for VectorStoreManager."""

    def setUp(self):
        """Set up test environment."""
        # Mock settings
        settings.DJANGO_LLM_SETTINGS = {
            "DEFAULT_EMBEDDING_MODEL": {
                "provider": "openai",
                "name": "text-embedding-ada-002",
            },
            "OPENAI_API_KEY": "test-key",
            "VECTOR_STORE": {"PGVECTOR_COLLECTION_NAME": "test_collection"},
        }
        settings.DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": "test_db",
                "USER": "test_user",
                "PASSWORD": "test_pass",
                "HOST": "localhost",
                "PORT": "5432",
            }
        }

    @patch("django_chain.services.vector_store_manager.OpenAIEmbeddings")
    def test_get_embeddings_model(self, mock_embeddings):
        """Test getting embeddings model."""
        # Mock the embeddings class
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        # Get embeddings model
        model = VectorStoreManager.get_embeddings_model()

        # Verify
        mock_embeddings.assert_called_once_with(
            openai_api_key="test-key", model="text-embedding-ada-002"
        )
        assert model == mock_instance

    @patch("django_chain.services.vector_store_manager.PGVector")
    def test_get_pgvector_store(self, mock_pgvector):
        """Test getting PGVector store."""
        # Mock the PGVector class
        mock_instance = MagicMock()
        mock_pgvector.return_value = mock_instance

        # Get vector store
        store = VectorStoreManager.get_pgvector_store()

        # Verify
        mock_pgvector.assert_called_once()
        assert store == mock_instance

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.get_pgvector_store")
    def test_add_documents(self, mock_get_store):
        """Test adding documents."""
        # Mock vector store
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Test data
        texts = ["doc1", "doc2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        # Add documents
        VectorStoreManager.add_documents(texts=texts, metadatas=metadatas)

        # Verify
        mock_store.add_texts.assert_called_once_with(texts=texts, metadatas=metadatas)

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.get_pgvector_store")
    def test_retrieve_documents(self, mock_get_store):
        """Test retrieving documents."""
        # Mock vector store and documents
        mock_store = MagicMock()
        mock_docs = [
            MagicMock(page_content="doc1", metadata={"source": "test1"}),
            MagicMock(page_content="doc2", metadata={"source": "test2"}),
        ]
        mock_store.similarity_search.return_value = mock_docs
        mock_get_store.return_value = mock_store

        # Retrieve documents
        docs = VectorStoreManager.retrieve_documents(query="test", k=2)

        # Verify
        mock_store.similarity_search.assert_called_once_with(query="test", k=2)
        assert len(docs) == 2
        assert docs[0].page_content == "doc1"
        assert docs[1].page_content == "doc2"

    def test_get_embeddings_model_no_api_key(self):
        """Test getting embeddings model without API key."""
        # Remove API key from settings
        settings.DJANGO_LLM_SETTINGS["OPENAI_API_KEY"] = None

        # Verify error
        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreManager.get_embeddings_model()
        assert "OPENAI_API_KEY is not configured" in str(exc_info.value)

    def test_get_embeddings_model_unsupported_provider(self):
        """Test getting embeddings model with unsupported provider."""
        # Set unsupported provider
        settings.DJANGO_LLM_SETTINGS["DEFAULT_EMBEDDING_MODEL"]["provider"] = "unsupported"

        # Verify error
        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreManager.get_embeddings_model()
        assert "Unsupported embedding provider" in str(exc_info.value)


@pytest.mark.django_db()
def test_vector_store_initialization() -> None:
    """Test vector store initialization."""
    manager = VectorStoreManager()
    assert manager is not None


@pytest.mark.skip()
@pytest.mark.django_db()
def test_embeddings_model() -> None:
    """Test embeddings model."""
    manager = VectorStoreManager()
    model = manager.get_embeddings_model()
    assert model is not None


@pytest.mark.skip()
@pytest.mark.django_db()
def test_pgvector_store() -> None:
    """Test pgvector store."""
    manager = VectorStoreManager()
    store = manager.get_pgvector_store()
    assert store is not None


@pytest.mark.skip()
@pytest.mark.django_db()
def test_document_operations() -> None:
    """Test document operations."""
    manager = VectorStoreManager()

    # Test adding documents
    documents = [
        {"content": "Test document 1", "metadata": {"source": "test1"}},
        {"content": "Test document 2", "metadata": {"source": "test2"}},
    ]
    manager.add_documents(documents)

    # Test retrieving documents
    results = manager.retrieve_documents("Test document", k=2)
    assert len(results) == 2
    assert results[0]["page_content"] == "Test document 1"
    assert results[1]["page_content"] == "Test document 2"


@pytest.mark.skip()
@pytest.mark.django_db()
def test_similarity_search() -> None:
    """Test similarity search."""
    manager = VectorStoreManager()

    # Add test documents
    documents = [
        {"content": "Python programming language", "metadata": {"source": "python"}},
        {"content": "JavaScript programming language", "metadata": {"source": "js"}},
    ]
    manager.add_documents(documents)

    # Test similarity search
    results = manager.retrieve_documents("Python", k=1)
    assert len(results) == 1
    assert "Python" in results[0]["page_content"]


@pytest.mark.skip()
@pytest.mark.django_db()
def test_metadata_filtering() -> None:
    """Test metadata filtering."""
    manager = VectorStoreManager()

    # Add test documents with metadata
    documents = [
        {"content": "Document 1", "metadata": {"category": "A", "source": "test1"}},
        {"content": "Document 2", "metadata": {"category": "B", "source": "test2"}},
    ]
    manager.add_documents(documents)

    # Test metadata filtering
    results = manager.retrieve_documents("Document", k=2, filter={"category": "A"})
    assert len(results) == 1
    assert results[0]["metadata"]["category"] == "A"

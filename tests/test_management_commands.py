"""
Tests for management commands.
"""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from django.core.management import call_command
from django.test import TestCase

from django_chain.services.vector_store_manager import VectorStoreManager


class TestManageVectorStoreCommand(TestCase):
    """Test cases for manage_vector_store command."""

    def setUp(self):
        """Set up test environment."""
        self.out = StringIO()
        self.err = StringIO()

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.add_documents")
    def test_add_from_file(self, mock_add_documents):
        """Test adding documents from file."""
        # Create test file
        test_data = [
            {"text": "doc1", "metadata": {"source": "test1"}},
            {"text": "doc2", "metadata": {"source": "test2"}},
        ]
        test_file = Path("test_docs.json")
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        try:
            # Run command
            call_command(
                "manage_vector_store",
                "add",
                file=str(test_file),
                stdout=self.out,
                stderr=self.err,
            )

            # Verify
            mock_add_documents.assert_called_once_with(
                texts=["doc1", "doc2"],
                metadatas=[{"source": "test1"}, {"source": "test2"}],
            )
            assert "Successfully added 2 documents" in self.out.getvalue()

        finally:
            # Clean up
            test_file.unlink()

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.add_documents")
    def test_add_single_document(self, mock_add_documents):
        """Test adding a single document."""
        # Run command
        call_command(
            "manage_vector_store",
            "add",
            text="test document",
            metadata='{"source": "test"}',
            stdout=self.out,
            stderr=self.err,
        )

        # Verify
        mock_add_documents.assert_called_once_with(
            texts=["test document"], metadatas=[{"source": "test"}]
        )
        assert "Successfully added 1 document" in self.out.getvalue()

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.retrieve_documents")
    def test_search(self, mock_retrieve_documents):
        """Test searching documents."""
        # Mock documents
        mock_docs = [
            MagicMock(page_content="doc1", metadata={"source": "test1"}),
            MagicMock(page_content="doc2", metadata={"source": "test2"}),
        ]
        mock_retrieve_documents.return_value = mock_docs

        # Run command
        call_command(
            "manage_vector_store",
            "search",
            "test query",
            limit=2,
            stdout=self.out,
            stderr=self.err,
        )

        # Verify
        mock_retrieve_documents.assert_called_once_with(query="test query", k=2)
        output = self.out.getvalue()
        assert "Found 2 documents" in output
        assert "doc1" in output
        assert "doc2" in output
        assert "test1" in output
        assert "test2" in output

    @pytest.mark.skip()
    @patch("django_chain.services.vector_store_manager.VectorStoreManager.get_pgvector_store")
    def test_clear(self, mock_get_store):
        """Test clearing vector store."""
        # Mock vector store
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Run command with force flag
        call_command("manage_vector_store", "clear", force=True, stdout=self.out, stderr=self.err)

        # Verify
        mock_store.delete_collection.assert_called_once()
        assert "Successfully cleared vector store" in self.out.getvalue()

    @pytest.mark.skip()
    def test_add_no_input(self):
        """Test add command with no input."""
        # Run command
        with pytest.raises(SystemExit):
            call_command("manage_vector_store", "add", stdout=self.out, stderr=self.err)

        # Verify error message
        assert "Either --file or --text must be specified" in self.err.getvalue()

    @pytest.mark.skip()
    def test_add_invalid_metadata(self):
        """Test add command with invalid metadata."""
        # Run command
        with pytest.raises(SystemExit):
            call_command(
                "manage_vector_store",
                "add",
                text="test document",
                metadata="invalid json",
                stdout=self.out,
                stderr=self.err,
            )

        # Verify error message
        assert "Invalid metadata JSON" in self.err.getvalue()

    @pytest.mark.skip()
    def test_add_invalid_file_format(self):
        """Test add command with invalid file format."""
        # Create test file with invalid format
        test_file = Path("test_docs.json")
        with open(test_file, "w") as f:
            f.write("invalid json")

        try:
            # Run command
            with pytest.raises(SystemExit):
                call_command(
                    "manage_vector_store",
                    "add",
                    file=str(test_file),
                    stdout=self.out,
                    stderr=self.err,
                )

            # Verify error message
            assert "Error reading file" in self.err.getvalue()

        finally:
            # Clean up
            test_file.unlink()


@pytest.mark.skip()
@pytest.mark.django_db()
def test_manage_vector_store_command() -> None:
    """Test manage_vector_store command."""
    # Test adding documents
    call_command(
        "manage_vector_store",
        "add",
        "--file",
        "test_documents.json",
        "--content-field",
        "text",
        "--metadata-fields",
        "source,category",
    )

    # Test retrieving documents
    manager = VectorStoreManager()
    results = manager.retrieve_documents("test query", k=5)
    assert len(results) > 0


@pytest.mark.skip()
@pytest.mark.django_db()
def test_manage_vector_store_command_with_invalid_file() -> None:
    """Test manage_vector_store command with invalid file."""
    with pytest.raises(FileNotFoundError):
        call_command(
            "manage_vector_store",
            "add",
            "--file",
            "nonexistent.json",
            "--content-field",
            "text",
        )


@pytest.mark.skip()
@pytest.mark.django_db()
def test_manage_vector_store_command_with_invalid_content_field() -> None:
    """Test manage_vector_store command with invalid content field."""
    with pytest.raises(ValueError):
        call_command(
            "manage_vector_store",
            "add",
            "--file",
            "test_documents.json",
            "--content-field",
            "nonexistent",
        )


@pytest.mark.skip()
@pytest.mark.django_db()
def test_manage_vector_store_command_with_invalid_metadata_fields() -> None:
    """Test manage_vector_store command with invalid metadata fields."""
    with pytest.raises(ValueError):
        call_command(
            "manage_vector_store",
            "add",
            "--file",
            "test_documents.json",
            "--content-field",
            "text",
            "--metadata-fields",
            "nonexistent",
        )

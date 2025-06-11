"""
Management command for managing vector store operations.
"""

import json
import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from django_chain.services.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Command for managing vector store operations."""

    help = """
    Manage vector store operations.

    Commands:
        add: Add documents to the vector store
        search: Search for documents in the vector store
        clear: Clear all documents from the vector store
    """

    def add_arguments(self, parser):
        """Add command arguments."""
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Add command
        add_parser = subparsers.add_parser("add", help="Add documents to the vector store")
        add_parser.add_argument(
            "--file", type=str, help="Path to a JSON file containing documents to add"
        )
        add_parser.add_argument("--text", type=str, help="Text to add as a document")
        add_parser.add_argument(
            "--metadata",
            type=str,
            help="JSON string containing metadata for the document",
        )

        # Search command
        search_parser = subparsers.add_parser("search", help="Search for documents")
        search_parser.add_argument("query", type=str, help="Search query")
        search_parser.add_argument(
            "--limit", type=int, default=4, help="Maximum number of results to return"
        )

        # Clear command
        clear_parser = subparsers.add_parser("clear", help="Clear all documents")
        clear_parser.add_argument(
            "--force", action="store_true", help="Force clearing without confirmation"
        )

    def handle(self, *args: Any, **options: dict[str, Any]) -> None:
        """Handle the command."""
        command = options.get("command")
        if not command:
            raise CommandError("No command specified")

        if command == "add":
            self.handle_add(**options)
        elif command == "search":
            self.handle_search(**options)
        elif command == "clear":
            self.handle_clear(**options)
        else:
            raise CommandError(f"Unknown command: {command}")

    def handle_add(self, **options: dict[str, Any]) -> None:
        """Handle the add command."""
        file_path = options.get("file")
        text = options.get("text")
        metadata_str = options.get("metadata")

        if not file_path and not text:
            raise CommandError("Either --file or --text must be specified")

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        if file_path:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                texts.append(item.get("text", ""))
                                metadatas.append(item.get("metadata", {}))
                            else:
                                texts.append(str(item))
                                metadatas.append({})
                    else:
                        raise CommandError("File must contain a list of documents")
            except Exception as e:
                raise CommandError(f"Error reading file: {e!s}")

        if text:
            texts.append(text)
            metadata = {}
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    raise CommandError("Invalid metadata JSON")
            metadatas.append(metadata)

        try:
            VectorStoreManager.add_documents(texts=texts, metadatas=metadatas)
            self.stdout.write(
                self.style.SUCCESS(f"Successfully added {len(texts)} documents to vector store")
            )
        except Exception as e:
            raise CommandError(f"Error adding documents: {e!s}")

    def handle_search(self, **options: dict[str, Any]) -> None:
        """Handle the search command."""
        query = options["query"]
        limit = options.get("limit", 4)

        try:
            docs = VectorStoreManager.retrieve_documents(query=query, k=limit)
            self.stdout.write(self.style.SUCCESS(f"Found {len(docs)} documents:"))
            for i, doc in enumerate(docs, 1):
                self.stdout.write(f"\n{i}. {doc.page_content}")
                if doc.metadata:
                    self.stdout.write(f"   Metadata: {json.dumps(doc.metadata, indent=2)}")
        except Exception as e:
            raise CommandError(f"Error searching documents: {e!s}")

    def handle_clear(self, **options: dict[str, Any]) -> None:
        """Handle the clear command."""
        force = options.get("force", False)

        if not force:
            confirm = input("Are you sure you want to clear all documents? [y/N] ")
            if confirm.lower() != "y":
                self.stdout.write("Operation cancelled")
                return

        try:
            vectorstore = VectorStoreManager.get_pgvector_store()
            vectorstore.delete_collection()
            self.stdout.write(self.style.SUCCESS("Successfully cleared vector store"))
        except Exception as e:
            raise CommandError(f"Error clearing vector store: {e!s}")

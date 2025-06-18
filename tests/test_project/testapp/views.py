"""
Test views for integration testing.
"""

import json
from typing import Any

from django.core import serializers
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django_chain.memory import get_langchain_memory, save_messages_to_session
from django_chain.models import LLMChain
from django_chain.models import ChatSession
from django_chain.services.chain_executor import ChainExecutor
from django_chain.services.llm_client import LLMClient
from django_chain.services.vector_store_manager import VectorStoreManager

from tests.test_project.testapp.models import TestDocument, TestPrompt

from django_chain.views import PromptView


class TestAppPrompts(PromptView):
    # TODO: Add post, update, delete
    def get(self, request, prompt_id=None, *args, **kwargs) -> JsonResponse:
        if prompt_id is None:
            prompts = TestPrompt.objects.all().values(
                "id",
                "guid",
                "name",
                "template",
                "input_variables",
                "optional_variables",
                "created_at",
            )
            return JsonResponse(list(prompts), safe=False)

        prompt = (
            TestPrompt.objects.filter(id=prompt_id)
            .all()
            .values(
                "id",
                "guid",
                "name",
                "template",
                "input_variables",
                "optional_variables",
                "created_at",
            )
        )
        return JsonResponse(list(prompt), safe=False)


@require_http_methods(["POST"])
def test_llm_call(request) -> JsonResponse:
    """Test a simple LLM call."""
    try:
        llm = LLMClient.get_chat_model()
        response = llm.invoke("Hello, how are you?")
        return JsonResponse({"status": "success", "response": response.content})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@require_http_methods(["POST"])
def test_chain_execution(request) -> JsonResponse:
    """Test chain execution."""
    try:
        # Create a test chain
        chain = LLMChain.objects.create(
            name="test_chain",
            prompt_template="Hello, {name}!",
            model_name="fake-model",
            input_variables=["name"],
        )

        # Execute the chain
        executor = ChainExecutor()
        result = executor.execute_chain(chain_id=chain.id, input_data={"name": "Test User"})

        return JsonResponse({"status": "success", "response": result})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@require_http_methods(["POST"])
def test_chat_session(request) -> JsonResponse:
    """Test chat session creation and message handling."""
    try:
        # Create a chat session
        session = ChatSession.objects.create(
            title="Test Chat", llm_config={"model_name": "fake-model"}
        )

        # Get memory for the session
        memory = get_langchain_memory(session)

        # Add a test message
        save_messages_to_session(session=session, messages=[{"type": "human", "content": "Hello!"}])

        return JsonResponse(
            {
                "status": "success",
                "session_id": session.id,
                "message_count": session.messages.count(),
            }
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@require_http_methods(["POST"])
def test_vector_store(request) -> JsonResponse:
    """Test vector store operations."""
    try:
        # Create a test document
        doc = TestDocument.objects.create(
            title="Test Document", content="This is a test document for vector store."
        )

        # Add to vector store
        VectorStoreManager.add_documents(
            texts=[doc.content], metadatas=[{"title": doc.title, "id": doc.id}]
        )

        # Search in vector store
        results = VectorStoreManager.retrieve_documents(query="test document", k=1)

        return JsonResponse(
            {
                "status": "success",
                "document_id": doc.id,
                "search_results": [doc.page_content for doc in results],
            }
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def test_chat_view(request: Any) -> JsonResponse:
    """Test view for chat functionality."""
    try:
        data = json.loads(request.body)
        message = data.get("message")
        session_id = data.get("session_id")

        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)

        client = LLMClient()
        response = client.chat(message, session_id)

        return JsonResponse(response)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def test_vector_search_view(request: Any) -> JsonResponse:
    """Test view for vector search functionality."""
    try:
        data = json.loads(request.body)
        query = data.get("query")
        k = data.get("k", 5)

        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)

        manager = VectorStoreManager()
        results = manager.retrieve_documents(query, k)

        return JsonResponse({"results": results})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def test_document_view(request: Any) -> JsonResponse:
    """Test view for document operations."""
    try:
        data = json.loads(request.body)
        title = data.get("title")
        content = data.get("content")

        if not title or not content:
            return JsonResponse({"error": "Title and content are required"}, status=400)

        doc = TestDocument.objects.create(title=title, content=content)

        return JsonResponse({"id": doc.id, "title": doc.title, "content": doc.content})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def test_document_list_view(request: Any) -> JsonResponse:
    """Test view for listing documents."""
    try:
        docs = TestDocument.objects.all()
        return JsonResponse(
            {
                "documents": [
                    {"id": doc.id, "title": doc.title, "content": doc.content} for doc in docs
                ]
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

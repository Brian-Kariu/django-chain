"""
Test views for integration testing.
"""

import json
from typing import Any

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django_chain.memory import get_langchain_memory
from django_chain.memory import save_messages_to_session
from django_chain.models import ChatSession
from django_chain.services.llm_client import LLMClient
from django_chain.services.vector_store_manager import VectorStoreManager


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

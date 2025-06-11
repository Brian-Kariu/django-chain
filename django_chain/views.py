import json
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.generic import View

from .services.llm_client import LLMClient
from .services.vector_store_manager import VectorStoreManager


class LLMView(View):
    """
    A base view for handling LLM requests.
    """

    chain = None  # type: Optional[Any]

    def get_prompt_context(self, request: HttpRequest) -> dict[str, Any]:
        """
        Get the context for the prompt template.
        Override this method to provide custom context.
        """
        return {}

    def process_response(self, response: Any) -> Any:
        """
        Process the LLM response before sending it to the client.
        Override this method to customize the response format.
        """
        return response

    def get(self, request: HttpRequest) -> JsonResponse:
        """
        Handle GET requests by running the LLM chain.
        """
        if not self.chain:
            return JsonResponse({"error": "No chain configured"}, status=400)

        try:
            context = self.get_prompt_context(request)
            chain = self.chain.get_chain()
            response = chain.run(**context)
            processed_response = self.process_response(response)
            return JsonResponse({"response": processed_response})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def chat_view(request: Any) -> JsonResponse:
    """Handle chat requests."""
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
def vector_search_view(request: Any) -> JsonResponse:
    """Handle vector search requests."""
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

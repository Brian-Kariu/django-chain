import json
from typing import Any

from django.forms import model_to_dict
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.generic import View

from django_chain.models import Prompt

from .services.llm_client import LLMClient
from .services.vector_store_manager import VectorStoreManager

from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.db import transaction


def serialize_queryset(queryset):
    if len(queryset) == 0:
        return []
    return [model_to_dict(instance) for instance in queryset]


class PromptView(View):
    def post(self, request, *args, **kwargs):
        prompt_data = self.kwargs["data"]
        if prompt_data is None:
            return JsonResponse({"No data passed"})

        prompt = Prompt.object.create(**prompt_data)
        data = model_to_dict(prompt)
        return JsonResponse(data, safe=False)

    def get(self, request, *args, **kwargs):
        if len(kwargs) == 0:
            prompts = Prompt.objects.all()
            data = serialize_queryset(prompts)
            return JsonResponse(data, safe=False)

        prompt_id = self.kwargs["prompt_id"]
        prompt = Prompt.objects.filter(id=prompt_id).first()
        data = model_to_dict(prompt)
        return JsonResponse(data, safe=False)


def serialize_prompt(prompt):
    """
    Custom serializer for Prompt instances, including handling JSONField.
    """
    data = model_to_dict(
        prompt, exclude=["id"]
    )  # Exclude id from model_to_dict if UUID is handled separately
    data["id"] = str(prompt.id)  # Convert UUID to string
    # JSONField content is already Python dict/list, no special handling needed for serialization
    return data


@csrf_exempt
def prompt_list_create(request):
    """
    Handles GET to list prompts and POST to create new prompt versions.
    """
    if request.method == "GET":
        include_inactive = request.GET.get("include_inactive", "false").lower() == "true"
        name_filter = request.GET.get("name")

        prompts = Prompt.objects.all()
        if name_filter:
            prompts = prompts.filter(name__iexact=name_filter)
        if not include_inactive:
            prompts = prompts.filter(is_active=True)

        data = [serialize_prompt(p) for p in prompts]
        return JsonResponse(data, safe=False)

    elif request.method == "POST":
        try:
            request_data = json.loads(request.body)
            name = request_data.get("name")
            prompt_template = request_data.get("prompt_template")
            description = request_data.get("description", "")
            input_variables = request_data.get("input_variables")
            activate = request_data.get("activate", True)  # Default to activating new versions

            if not name or not prompt_template:
                return JsonResponse({"error": "Name and prompt_template are required."}, status=400)

            # Use a transaction to ensure atomicity for versioning logic
            with transaction.atomic():
                new_prompt = Prompt.create_new_version(
                    name=name,
                    prompt_template=prompt_template,
                    description=description,
                    input_variables=input_variables,
                    activate=activate,
                )
            return JsonResponse(serialize_prompt(new_prompt), status=201)  # 201 Created

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request body."}, status=400)
        except ValidationError as e:
            return JsonResponse({"error": e.message_dict}, status=400)
        except Exception as e:
            # Catch any other unexpected errors
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Method Not Allowed"}, status=405)


@csrf_exempt
def prompt_detail(request, pk):
    """
    Handles GET, PUT, DELETE for a specific prompt version by UUID.
    """
    try:
        # Use get_object_or_404 in real-world, but for vanilla, manual check.
        prompt = Prompt.objects.get(id=pk)
    except ObjectDoesNotExist:
        return JsonResponse({"error": "Prompt not found."}, status=404)
    except ValidationError:  # Catch UUID validation error if pk is not a valid UUID
        return JsonResponse({"error": "Invalid Prompt ID format."}, status=400)

    if request.method == "GET":
        return JsonResponse(serialize_prompt(prompt))

    elif request.method == "PUT":
        try:
            request_data = json.loads(request.body)
            # Allow updating description and input_variables directly.
            # Changes to prompt_template should ideally be new versions (handled by POST to list_create).
            prompt.description = request_data.get("description", prompt.description)
            prompt.input_variables = request_data.get("input_variables", prompt.input_variables)

            # If prompt_template is provided in PUT, it indicates a desire to modify current version's template.
            # This is less ideal for versioning, but acceptable if explicitly needed.
            # For strict versioning, client should use POST to create_new_version.
            if "prompt_template" in request_data:
                prompt.prompt_template = request_data["prompt_template"]

            prompt.full_clean()  # Validate updated fields
            prompt.save()
            return JsonResponse(serialize_prompt(prompt))

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request body."}, status=400)
        except ValidationError as e:
            return JsonResponse({"error": e.message_dict}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    elif request.method == "DELETE":
        prompt.delete()
        return JsonResponse(
            {"message": "Prompt deleted successfully."}, status=204
        )  # 204 No Content
    else:
        return JsonResponse({"error": "Method Not Allowed"}, status=405)


@csrf_exempt
def prompt_activate(request, pk):
    """
    Handles POST to activate a specific prompt version.
    """
    if request.method == "POST":
        try:
            prompt = Prompt.objects.get(id=pk)
        except ObjectDoesNotExist:
            return JsonResponse({"error": "Prompt not found."}, status=404)
        except ValidationError:
            return JsonResponse({"error": "Invalid Prompt ID format."}, status=400)

        try:
            with transaction.atomic():
                prompt.activate()  # Call the model method to handle activation logic
            return JsonResponse(serialize_prompt(prompt))
        except ValidationError as e:
            return JsonResponse({"error": e.message_dict}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Method Not Allowed"}, status=405)


@csrf_exempt
def prompt_deactivate(request, pk):
    """
    Handles POST to deactivate a specific prompt version.
    """
    if request.method == "POST":
        try:
            prompt = Prompt.objects.get(id=pk)
        except ObjectDoesNotExist:
            return JsonResponse({"error": "Prompt not found."}, status=404)
        except ValidationError:
            return JsonResponse({"error": "Invalid Prompt ID format."}, status=400)

        try:
            prompt.deactivate()  # Call the model method
            return JsonResponse(serialize_prompt(prompt))
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Method Not Allowed"}, status=405)


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

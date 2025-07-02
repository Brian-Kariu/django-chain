"""
Views for django-chain: API endpoints for prompt, workflow, user interaction, and LLM execution.

This module provides Django class-based and function-based views for managing prompts, workflows,
user interactions, and LLM-powered chat or vector search endpoints.

Typical usage example:
    urlpatterns = [
        path('api/', include('django_chain.urls')),
    ]
"""

import json
import uuid
from typing import Any

from django.conf import settings
from django.forms import model_to_dict
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views import View

from django_chain.models import InteractionLog, Prompt

from .services.llm_client import LLMClient
from .services.vector_store_manager import VectorStoreManager

from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.db import transaction, models

from django_chain.models import Prompt, Workflow, UserInteraction
from django_chain.utils.llm_client import (
    _execute_and_log_workflow_step,
    _to_serializable,
)

from django_chain.mixins import (
    JSONResponseMixin,
    ModelRetrieveMixin,
    ModelListMixin,
    ModelCreateMixin,
    ModelUpdateMixin,
    ModelDeleteMixin,
    ModelActivateDeactivateMixin,
)


def serialize_queryset(queryset):
    if len(queryset) == 0:
        return []
    return [instance.to_dict() for instance in queryset]


def serialize_user_interaction(user_interaction, include_logs=False):
    """
    Custom serializer for UserInteraction instances.
    Optionally includes related InteractionLog entries.
    """
    data = model_to_dict(user_interaction, exclude=["id", "workflow"])
    data["id"] = str(user_interaction.id)
    data["session_id"] = str(user_interaction.session_id) if user_interaction.session_id else None
    data["workflow"] = (
        {"id": str(user_interaction.workflow.id), "name": user_interaction.workflow.name}
        if user_interaction.workflow
        else None
    )

    data["input_data"] = user_interaction.input_data
    data["llm_output"] = user_interaction.llm_output

    return data


class PromptListCreateView(JSONResponseMixin, ModelListMixin, ModelCreateMixin, View):
    """
    View for listing and creating Prompt objects.

    GET: List all prompts (optionally filter by name or active status).
    POST: Create a new prompt version.
    """

    model_class = Prompt
    serializer_method = lambda view, p: p.to_dict()
    required_fields = ["name", "prompt_template"]

    def get(self, request, *args, **kwargs) -> JsonResponse:
        """
        List all prompts, optionally filtered by name or active status.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            JsonResponse: List of serialized prompts.
        """
        prompts = self.get_queryset(request)
        prompts = self.apply_list_filters(prompts, request)
        data = [self.serializer_method(p) for p in prompts]
        return self.render_json_response(data, safe=False)

    def post(self, request, *args, **kwargs) -> JsonResponse:
        """
        Create a new prompt version.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            JsonResponse: The created prompt or error message.
        """
        request_data = request.json_body
        try:
            name = request_data.get("name")
            prompt_template = request_data.get("prompt_template")
            input_variables = request_data.get("input_variables")
            activate = request_data.get("activate", True)

            with transaction.atomic():
                new_prompt = Prompt.create_new_version(
                    name=name,
                    prompt_template=prompt_template,
                    input_variables=input_variables,
                    activate=activate,
                )
            return self.render_json_response(self.serializer_method(new_prompt), status=201)
        except ValidationError as e:
            return self.json_error_response(e.message_dict, status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def apply_list_filters(self, queryset, request) -> models.QuerySet:
        """
        Apply filters to the queryset based on request parameters.

        Args:
            queryset (QuerySet): The queryset to filter.
            request (HttpRequest): The HTTP request object.

        Returns:
            QuerySet: The filtered queryset.
        """
        include_inactive = request.GET.get("include_inactive", "false").lower() == "true"
        name_filter = request.GET.get("name")

        if name_filter:
            queryset = queryset.filter(name__iexact=name_filter)
        if not include_inactive:
            queryset = queryset.filter(is_active=True)
        return queryset


class PromptDetailView(
    JSONResponseMixin, ModelRetrieveMixin, ModelUpdateMixin, ModelDeleteMixin, View
):
    """
    View for retrieving, updating, or deleting a single Prompt object.

    GET: Retrieve a prompt by primary key.
    PUT: Update a prompt's input variables or template.
    DELETE: Delete a prompt.
    """

    model_class = Prompt
    serializer_method = lambda view, p: p.to_dict()

    def get(self, request, pk: str, *args, **kwargs) -> JsonResponse:
        """
        Retrieve a prompt by primary key.

        Args:
            request (HttpRequest): The HTTP request object.
            pk (str): The primary key of the prompt.

        Returns:
            JsonResponse: The serialized prompt or error message.
        """
        prompt = self.get_object(pk)
        if prompt is None:
            return self.json_error_response("Prompt not found.", status=404)
        return self.render_json_response(self.serializer_method(prompt))

    def put(self, request, pk: str, *args, **kwargs) -> JsonResponse:
        """
        Update a prompt's input variables or template.

        Args:
            request (HttpRequest): The HTTP request object.
            pk (str): The primary key of the prompt.

        Returns:
            JsonResponse: The updated prompt or error message.
        """
        prompt = self.get_object(pk)
        if prompt is None:
            return self.json_error_response("Prompt not found.", status=404)

        request_data = request.json_body
        try:
            if "input_variables" in request_data:
                prompt.input_variables = request_data["input_variables"]
            if "prompt_template" in request_data:
                prompt.prompt_template = request_data["prompt_template"]

            prompt.full_clean()
            prompt.save()
            return self.render_json_response(self.serializer_method(prompt))
        except ValidationError as e:
            return self.json_error_response(e.message_dict, status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def delete(self, request, pk: str, *args, **kwargs) -> JsonResponse:
        """
        Delete a prompt by primary key.

        Args:
            request (HttpRequest): The HTTP request object.
            pk (str): The primary key of the prompt.

        Returns:
            JsonResponse: Success message or error message.
        """
        prompt = self.get_object(pk)
        if prompt is None:
            return self.json_error_response("Prompt not found.", status=404)

        try:
            self.delete_object(prompt)
            return self.render_json_response(
                {"message": "Prompt deleted successfully."}, status=204
            )
        except Exception as e:
            return self.json_error_response(str(e), status=500)


class PromptActivateDeactivateView(
    JSONResponseMixin, ModelRetrieveMixin, ModelActivateDeactivateMixin, View
):
    model_class = Prompt
    serializer_method = lambda view, p: p.to_dict()

    def post(self, request, pk, action, *args, **kwargs):
        return super().post(request, pk, action, *args, **kwargs)


class WorkflowListCreateView(JSONResponseMixin, ModelListMixin, ModelCreateMixin, View):
    model_class = Workflow
    serializer_method = lambda view, w: w.to_dict()
    required_fields = ["name", "workflow_definition"]

    def get(self, request, *args, **kwargs):
        workflows = self.get_queryset(request)
        workflows = self.apply_list_filters(workflows, request)
        data = [self.serializer_method(w) for w in workflows]
        return self.render_json_response(data, safe=False)

    def post(self, request, *args, **kwargs):
        request_data = request.json_body
        try:
            name = request_data.get("name")
            description = request_data.get("description", "")
            workflow_definition = request_data.get("workflow_definition")
            activate = request_data.get("activate", False)

            if not name or not workflow_definition:
                return self.json_error_response(
                    "Name and workflow_definition are required.", status=400
                )

            with transaction.atomic():
                workflow = Workflow(
                    name=name,
                    description=description,
                    workflow_definition=workflow_definition,
                    is_active=activate,
                )
                workflow.full_clean()
                workflow.save()

                if activate:
                    workflow.activate()
            return self.render_json_response(self.serializer_method(workflow), status=201)

        except ValidationError as e:
            return self.json_error_response(e.message_dict, status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def apply_list_filters(self, queryset, request):
        include_inactive = request.GET.get("include_inactive", "false").lower() == "true"
        name_filter = request.GET.get("name")

        if name_filter:
            queryset = queryset.filter(name__iexact=name_filter)
        if not include_inactive:
            queryset = queryset.filter(is_active=True)
        return queryset


class WorkflowDetailView(
    JSONResponseMixin, ModelRetrieveMixin, ModelUpdateMixin, ModelDeleteMixin, View
):
    model_class = Workflow
    serializer_method = lambda view, w: w.to_dict()

    def get(self, request, pk, *args, **kwargs):
        workflow = self.get_object(pk)
        if workflow is None:
            return self.json_error_response("Workflow not found.", status=404)
        return self.render_json_response(self.serializer_method(workflow))

    def put(self, request, pk, *args, **kwargs):
        workflow = self.get_object(pk)
        if workflow is None:
            return self.json_error_response("Workflow not found.", status=404)

        request_data = request.json_body
        try:
            workflow.description = request_data.get("description", workflow.description)
            workflow.workflow_definition = request_data.get(
                "workflow_definition", workflow.workflow_definition
            )
            workflow.full_clean()
            workflow.save()
            return self.render_json_response(self.serializer_method(workflow))
        except ValidationError as e:
            return self.json_error_response(e.message_dict, status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def delete(self, request, pk, *args, **kwargs):
        workflow = self.get_object(pk)
        if workflow is None:
            return self.json_error_response("Workflow not found.", status=404)

        try:
            self.delete_object(workflow)
            return self.render_json_response(
                {"message": "Workflow deleted successfully."}, status=204
            )
        except Exception as e:
            return self.json_error_response(str(e), status=500)


class InteractionLogListCreateView(JSONResponseMixin, ModelListMixin, ModelCreateMixin, View):
    model_class = InteractionLog
    serializer_method = lambda view, w: w.to_dict()
    required_fields = ["model_name", "provider"]

    def get(self, request, *args, **kwargs):
        logs = self.get_queryset(request)
        data = [self.serializer_method(w) for w in logs]
        return self.render_json_response(data, safe=False)


class InteractionLogDetailView(
    JSONResponseMixin, ModelRetrieveMixin, ModelUpdateMixin, ModelDeleteMixin, View
):
    model_class = InteractionLog
    serializer_method = lambda view, w: w.to_dict()

    def get(self, request, pk, *args, **kwargs):
        log = self.get_object(pk)
        if log is None:
            return self.json_error_response("Interaction Log not found.", status=404)
        return self.render_json_response(self.serializer_method(log))

    def delete(self, request, pk, *args, **kwargs):
        log = self.get_object(pk)
        if log is None:
            return self.json_error_response("Log not found.", status=404)

        try:
            self.delete_object(log)
            return self.render_json_response({"message": "Log deleted successfully."}, status=204)
        except Exception as e:
            return self.json_error_response(str(e), status=500)


class WorkflowActivateDeactivateView(
    JSONResponseMixin, ModelRetrieveMixin, ModelActivateDeactivateMixin, View
):
    model_class = Workflow
    serializer_method = lambda view, w: w.to_dict()

    def post(self, request, pk, action, *args, **kwargs):
        return super().post(request, pk, action, *args, **kwargs)


class UserInteractionListView(JSONResponseMixin, ModelListMixin, View):
    model_class = UserInteraction
    serializer_method = serialize_user_interaction

    def get(self, request, *args, **kwargs):
        interactions = self.get_queryset(request).select_related("workflow")
        interactions = self.apply_list_filters(interactions, request)
        data = [self.serializer_method(ui) for ui in interactions]
        return self.render_json_response(data, safe=False)

    def apply_list_filters(self, queryset, request):
        workflow_id = request.GET.get("workflow_id")
        workflow_name = request.GET.get("workflow_name")
        user_identifier = request.GET.get("user_identifier")
        session_id = request.GET.get("session_id")
        status = request.GET.get("status")

        if workflow_id:
            try:
                queryset = queryset.filter(workflow__id=uuid.UUID(workflow_id))
            except ValueError:
                raise ValidationError("Invalid workflow_id format.")

        if workflow_name:
            queryset = queryset.filter(workflow__name__iexact=workflow_name)

        if user_identifier:
            queryset = queryset.filter(user_identifier__iexact=user_identifier)

        if session_id:
            try:
                queryset = queryset.filter(session_id=uuid.UUID(session_id))
            except ValueError:
                raise ValidationError("Invalid session_id format.")

        if status:
            queryset = queryset.filter(status__iexact=status)

        return queryset


class UserInteractionDetailView(JSONResponseMixin, ModelRetrieveMixin, View):
    model_class = UserInteraction
    serializer_method = serialize_user_interaction

    def get(self, request, pk, *args, **kwargs):
        user_interaction = self.get_object(pk)
        if user_interaction is None:
            return self.json_error_response("User Interaction not found.", status=404)

        data = self.serializer_method(user_interaction, include_logs=True)
        return self.render_json_response(data)


class ExecuteWorkflowView(JSONResponseMixin, View):
    def post(self, request, name, *args, **kwargs):
        try:
            request_data = request.json_body
            input_data = request_data.get("input", {})
            execution_method = request_data.get("execution_method", "INVOKE")
            execution_config = request_data.get("execution_config", {})

            if not isinstance(input_data, dict):
                raise ValidationError("Input data must be a JSON object.")

            workflow_record = Workflow.objects.get(name__iexact=name, is_active=True)

            global_llm_config = getattr(settings, "DJANGO_LLM_SETTINGS", {})

            workflow_chain = workflow_record.to_langchain_chain(
                llm_config=global_llm_config, log="true"
            )

            response = _execute_and_log_workflow_step(
                workflow_chain, input_data, execution_method, execution_config
            )

        except ObjectDoesNotExist:
            overall_error_message = f"No active workflow found with name: {name}"
            return self.json_error_response(overall_error_message, status=404)
        except json.JSONDecodeError:
            overall_error_message = "Invalid JSON in request body."
            return self.json_error_response(overall_error_message, status=400)
        except ValidationError as e:
            return self.json_error_response(str(e), status=400)
        except Exception as e:
            overall_error_message = f"An unexpected error occurred: {str(e)}"
            return self.json_error_response(overall_error_message, status=500)

        return self.render_json_response(
            {
                "workflow_name": workflow_record.name,
                "input_received": input_data,
                "output": _to_serializable(response),
            }
        )


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

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

from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.db import transaction, models
from django.forms import model_to_dict
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views import View
from django.utils import timezone

from django_chain.config import app_settings
from django_chain.exceptions import PromptValidationError
from django_chain.models import InteractionLog, Prompt, Workflow, ChatSession, ChatHistory
from django_chain.utils.llm_client import (
    _execute_and_log_workflow_step,
    _to_serializable,
    create_llm_chat_client,
)

from .services.vector_store_manager import VectorStoreManager

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


class PromptActivateDeactivateView(ModelActivateDeactivateMixin, View):
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
            name = request_data.pop("name")
            description = request_data.pop("description", "")
            workflow_definition = request_data.pop("workflow_definition")
            prompt_id = request_data.pop("prompt")
            activate = request_data.pop("activate", False)

            prompt_instance = Prompt.objects.get(id=prompt_id)

            if not name or not workflow_definition:
                return self.json_error_response(
                    "Name and workflow_definition are required.", status=400
                )

            with transaction.atomic():
                workflow = Workflow(
                    name=name,
                    description=description,
                    prompt=prompt_instance,
                    workflow_definition=workflow_definition,
                    is_active=activate,
                )
                workflow.full_clean()
                workflow.save()

                if activate:
                    workflow.activate()
            return self.render_json_response(self.serializer_method(workflow), status=201)

        except ValidationError as e:
            return self.json_error_response(str(e), status=400)
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
            prompt_instance = Prompt.objects.get(id=request_data.get("prompt", None))
            workflow.prompt = prompt_instance
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


class WorkflowActivateDeactivateView(ModelActivateDeactivateMixin, View):
    model_class = Workflow
    serializer_method = lambda view, w: w.to_dict()

    def post(self, request, pk, action, *args, **kwargs):
        return super().post(request, pk, action, *args, **kwargs)


class ExecuteWorkflowView(JSONResponseMixin, View):
    """
    Execute a workflow with optional chat session integration.

    When session_id is provided, integrates with ChatSession and ChatHistory models
    to maintain conversation context and log all interactions.
    """

    def get_or_create_chat_session(self, session_id, workflow, user=None):
        """Get or create a chat session."""
        if isinstance(session_id, str):
            try:
                session_id = uuid.UUID(session_id)
            except ValueError:
                session_id = uuid.uuid4()

        try:
            if user and hasattr(user, "is_authenticated") and user.is_authenticated:
                session = ChatSession.objects.get(session_id=session_id, user=user)
            else:
                session = ChatSession.objects.get(session_id=session_id, user__isnull=True)
        except ChatSession.DoesNotExist:
            session_data = {
                "session_id": session_id,
                "workflow": workflow,
                "title": f"Chat with {workflow.name}",
                "is_active": True,
            }
            if user and hasattr(user, "is_authenticated") and user.is_authenticated:
                session_data["user"] = user
            session = ChatSession.objects.create(**session_data)

        return session

    def log_user_message(self, session, message_content):
        """Log user message to chat history."""
        last_message = session.messages.order_by("-order").first()
        next_order = (last_message.order + 1) if last_message else 0

        user_message = ChatHistory.objects.create(
            session=session,
            content=message_content,
            role="USER",
            order=next_order,
        )
        return user_message

    def log_assistant_message(self, session, response_content, token_count=None):
        """Log assistant response to chat history."""
        last_message = session.messages.order_by("-order").first()
        next_order = (last_message.order + 1) if last_message else 0

        assistant_message = ChatHistory.objects.create(
            session=session,
            content=response_content,
            role="ASSISTANT",
            order=next_order,
            token_count=token_count,
        )
        return assistant_message

    def post(self, request, name, *args, **kwargs):
        try:
            request_data = json.loads(request.body) if request.body else {}
            input_data = request_data.get("input", {})
            execution_method = request_data.get("execution_method", "invoke")
            execution_config = request_data.get("execution_config", {})
            session_id = request_data.get("session_id")

            if session_id:
                if isinstance(session_id, str):
                    try:
                        session_id = uuid.UUID(session_id)
                    except ValueError:
                        return self.json_error_response(
                            f"Invalid session_id format: {session_id}. Must be a valid UUID.",
                            status=400,
                        )

            try:
                workflow_record = Workflow.objects.get(name=name)
            except Workflow.DoesNotExist:
                return self.json_error_response(f"Workflow '{name}' not found", status=404)

            global_llm_config = getattr(app_settings, "DJANGO_LLM_SETTINGS", {})

            if session_id:
                user = getattr(request, "user", None)

                authenticated_user = (
                    user
                    if (user and hasattr(user, "is_authenticated") and user.is_authenticated)
                    else None
                )

                chat_session = self.get_or_create_chat_session(
                    session_id, workflow_record, authenticated_user
                )

                message_content = (
                    input_data.get("message") or input_data.get("input") or str(input_data)
                )
                if message_content and message_content != "{}":
                    user_message = self.log_user_message(chat_session, message_content)

                history = request_data.pop("history", "chat_history")
                execution_config["configurable"] = {"session_id": str(session_id)}
                workflow_chain = workflow_record.to_langchain_chain(
                    llm_config=global_llm_config,
                    log="true",
                    session_id=str(session_id),
                    chat_input=input_data.get("chat_input", "input"),
                    history=history,
                    user=authenticated_user,
                )

                response = _execute_and_log_workflow_step(
                    workflow_chain=workflow_chain,
                    current_input=input_data,
                    execution_method=execution_method,
                    execution_config=execution_config,
                )

                response_content = str(response) if response else "No response"
                assistant_message = self.log_assistant_message(
                    chat_session,
                    response_content,
                    token_count=getattr(response, "usage", {}).get("total_tokens")
                    if hasattr(response, "usage")
                    else None,
                )

                chat_session.save()

                response_data = {
                    "workflow_name": workflow_record.name,
                    "session_id": str(session_id),
                    "input_received": input_data,
                    "output": _to_serializable(response),
                    "chat_session": {
                        "id": chat_session.id,
                        "title": chat_session.title,
                        "message_count": chat_session.messages.filter(is_deleted=False).count(),
                    },
                    "messages": {
                        "user_message_id": user_message.id if user_message else None,
                        "assistant_message_id": assistant_message.id,
                    },
                }

                return self.render_json_response(response_data)

            else:
                workflow_chain = workflow_record.to_langchain_chain(
                    llm_config=global_llm_config, log="true"
                )

                response = _execute_and_log_workflow_step(
                    workflow_chain=workflow_chain,
                    current_input=input_data,
                    execution_method=execution_method,
                    execution_config=execution_config,
                )

                response_data = {
                    "workflow_name": workflow_record.name,
                    "input_received": input_data,
                    "output": _to_serializable(response),
                }

                return self.render_json_response(response_data)

        except json.JSONDecodeError:
            return self.json_error_response("Invalid JSON in request body", status=400)
        except Exception as e:
            return self.json_error_response(f"Workflow execution failed: {str(e)}", status=500)


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

        provider = app_settings.DEFAULT_LLM_PROVIDER
        chat_model = create_llm_chat_client(provider)

        if chat_model is None:
            return JsonResponse({"error": "Failed to initialize chat model"}, status=500)

        response = chat_model.invoke(message)

        if hasattr(response, "content"):
            response_content = response.content
        else:
            response_content = str(response)

        return JsonResponse({"response": response_content, "session_id": session_id})
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


class ChatSessionListCreateView(JSONResponseMixin, ModelListMixin, ModelCreateMixin, View):
    """
    List all chat sessions or create a new chat session.

    GET: Returns list of chat sessions (active by default)
    POST: Creates a new chat session
    """

    model = ChatSession

    def get_queryset(self):
        """Get queryset of chat sessions, filtered by user and active status."""
        queryset = self.model.objects.select_related("workflow", "user")

        user = getattr(self.request, "user", None)
        if user and user.is_authenticated:
            queryset = queryset.filter(user=user)

        include_archived = self.request.GET.get("include_archived", "false").lower() == "true"
        if not include_archived:
            queryset = queryset.filter(is_active=True)

        return queryset.order_by("-updated_at")

    def get(self, request, *args, **kwargs):
        """List chat sessions."""
        try:
            sessions = self.get_queryset()

            page = int(request.GET.get("page", 1))
            page_size = int(request.GET.get("page_size", 20))
            start = (page - 1) * page_size
            end = start + page_size

            sessions_data = []
            for session in sessions[start:end]:
                session_data = {
                    "id": session.id,
                    "session_id": session.session_id,
                    "title": session.title,
                    "workflow_id": session.workflow.id if session.workflow else None,
                    "workflow_name": session.workflow.name if session.workflow else None,
                    "is_active": session.is_active,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "message_count": session.messages.filter(is_deleted=False).count(),
                }
                sessions_data.append(session_data)

            return self.render_json_response(
                {
                    "sessions": sessions_data,
                    "page": page,
                    "page_size": page_size,
                    "total": sessions.count(),
                }
            )

        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def post(self, request, *args, **kwargs):
        """Create a new chat session."""
        try:
            data = json.loads(request.body)

            session_id = data.get("session_id")
            if session_id:
                if isinstance(session_id, str):
                    try:
                        session_id = uuid.UUID(session_id)
                    except ValueError:
                        return self.json_error_response(
                            f"Invalid session_id format: {session_id}. Must be a valid UUID.",
                            status=400,
                        )
            else:
                session_id = uuid.uuid4()

            workflow = None
            workflow_id = data.get("workflow_id")
            if workflow_id:
                try:
                    workflow = Workflow.objects.get(id=workflow_id)
                except Workflow.DoesNotExist:
                    return self.json_error_response(
                        f"Workflow with id {workflow_id} not found", status=400
                    )

            session_data = {
                "session_id": session_id,
                "title": data.get("title", ""),
                "workflow": workflow,
                "is_active": data.get("is_active", True),
            }

            user = getattr(request, "user", None)
            if user and user.is_authenticated:
                session_data["user"] = user

            session = ChatSession.objects.create(**session_data)

            return self.render_json_response(
                {
                    "id": session.id,
                    "session_id": str(session.session_id),
                    "title": session.title,
                    "workflow_id": session.workflow.id if session.workflow else None,
                    "is_active": session.is_active,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                },
                status=201,
            )

        except json.JSONDecodeError:
            return self.json_error_response("Invalid JSON", status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)


class ChatSessionDetailView(
    JSONResponseMixin, ModelRetrieveMixin, ModelUpdateMixin, ModelDeleteMixin, View
):
    """
    Retrieve, update, or delete a specific chat session.

    GET: Returns chat session details with message history
    PUT/PATCH: Updates chat session
    DELETE: Archives chat session (soft delete)
    """

    model = ChatSession

    def get_object(self, session_id):
        """Get chat session by session_id."""
        try:
            if isinstance(session_id, str):
                try:
                    session_id = uuid.UUID(session_id)
                except ValueError:
                    return None

            user = getattr(self.request, "user", None)
            queryset = self.model.objects.select_related("workflow", "user")

            if user and user.is_authenticated:
                return queryset.get(session_id=session_id, user=user)
            else:
                return queryset.get(session_id=session_id, user__isnull=True)

        except self.model.DoesNotExist:
            return None

    def get(self, request, session_id, *args, **kwargs):
        """Get chat session with message history."""
        try:
            session = self.get_object(session_id)
            if not session:
                return self.json_error_response("Chat session not found", status=404)

            include_deleted = request.GET.get("include_deleted", "false").lower() == "true"
            if include_deleted:
                messages = session.messages.all().order_by("timestamp", "order")
            else:
                messages = session.messages.filter(is_deleted=False).order_by("timestamp", "order")

            messages_data = []
            for message in messages:
                messages_data.append(
                    {
                        "id": message.id,
                        "content": message.content,
                        "role": message.role,
                        "timestamp": message.timestamp.isoformat(),
                        "token_count": message.token_count,
                        "order": message.order,
                        "is_deleted": message.is_deleted,
                    }
                )

            return self.render_json_response(
                {
                    "id": session.id,
                    "session_id": str(session.session_id),
                    "title": session.title,
                    "workflow_id": session.workflow.id if session.workflow else None,
                    "workflow_name": session.workflow.name if session.workflow else None,
                    "is_active": session.is_active,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "messages": messages_data,
                    "message_count": len([m for m in messages_data if not m["is_deleted"]]),
                }
            )

        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def put(self, request, session_id, *args, **kwargs):
        """Update chat session."""
        try:
            session = self.get_object(session_id)
            if not session:
                return self.json_error_response("Chat session not found", status=404)

            data = json.loads(request.body)

            if "title" in data:
                session.title = data["title"]
            if "is_active" in data:
                session.is_active = data["is_active"]
            if "workflow_id" in data:
                if data["workflow_id"]:
                    try:
                        workflow = Workflow.objects.get(id=data["workflow_id"])
                        session.workflow = workflow
                    except Workflow.DoesNotExist:
                        return self.json_error_response(
                            f"Workflow with id {data['workflow_id']} not found", status=400
                        )
                else:
                    session.workflow = None

            session.save()

            return self.render_json_response(
                {
                    "id": session.id,
                    "session_id": str(session.session_id),
                    "title": session.title,
                    "workflow_id": session.workflow.id if session.workflow else None,
                    "is_active": session.is_active,
                    "updated_at": session.updated_at.isoformat(),
                }
            )

        except json.JSONDecodeError:
            return self.json_error_response("Invalid JSON", status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def delete(self, request, session_id, *args, **kwargs):
        """Archive chat session (soft delete)."""
        try:
            session = self.get_object(session_id)
            if not session:
                return self.json_error_response("Chat session not found", status=404)

            session.archive()

            return self.render_json_response(
                {
                    "message": "Chat session archived successfully",
                    "session_id": str(session.session_id),
                    "is_active": session.is_active,
                }
            )

        except Exception as e:
            return self.json_error_response(str(e), status=500)


class ChatHistoryView(JSONResponseMixin, View):
    """
    Manage chat history messages within a session.

    GET: List messages for a session
    POST: Add a new message to a session
    DELETE: Soft delete a message
    """

    def get_session(self, session_id, user=None):
        """Get chat session by session_id."""
        try:
            if isinstance(session_id, str):
                try:
                    session_id = uuid.UUID(session_id)
                except ValueError:
                    return None

            queryset = ChatSession.objects.select_related("workflow")
            if user and user.is_authenticated:
                return queryset.get(session_id=session_id, user=user)
            else:
                return queryset.get(session_id=session_id, user__isnull=True)
        except ChatSession.DoesNotExist:
            return None

    def get(self, request, session_id, *args, **kwargs):
        """Get chat history for a session."""
        try:
            user = getattr(request, "user", None)
            session = self.get_session(session_id, user)
            if not session:
                return self.json_error_response("Chat session not found", status=404)

            include_deleted = request.GET.get("include_deleted", "false").lower() == "true"
            if include_deleted:
                messages = session.messages.all().order_by("timestamp", "order")
            else:
                messages = session.messages.filter(is_deleted=False).order_by("timestamp", "order")

            messages_data = []
            for message in messages:
                messages_data.append(
                    {
                        "id": message.id,
                        "content": message.content,
                        "role": message.role,
                        "timestamp": message.timestamp.isoformat(),
                        "token_count": message.token_count,
                        "order": message.order,
                        "is_deleted": message.is_deleted,
                    }
                )

            return self.render_json_response(
                {
                    "session_id": str(session_id) if session_id else None,
                    "messages": messages_data,
                    "total": len(messages_data),
                }
            )

        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def post(self, request, session_id, *args, **kwargs):
        """Add a new message to the chat history."""
        try:
            user = getattr(request, "user", None)
            session = self.get_session(session_id, user)
            if not session:
                return self.json_error_response("Chat session not found", status=404)

            data = json.loads(request.body)
            content = data.get("content", "")
            role = data.get("role", "USER")

            if not content:
                return self.json_error_response("Message content is required", status=400)

            last_message = session.messages.order_by("-order").first()
            next_order = (last_message.order + 1) if last_message else 0

            message = ChatHistory.objects.create(
                session=session,
                content=content,
                role=role,
                order=next_order,
                token_count=data.get("token_count"),
            )

            session.save()

            return self.render_json_response(
                {
                    "id": message.id,
                    "content": message.content,
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    "token_count": message.token_count,
                    "order": message.order,
                    "is_deleted": message.is_deleted,
                },
                status=201,
            )

        except json.JSONDecodeError:
            return self.json_error_response("Invalid JSON", status=400)
        except Exception as e:
            return self.json_error_response(str(e), status=500)

    def delete(self, request, session_id, message_id, *args, **kwargs):
        """Soft delete a message."""
        try:
            user = getattr(request, "user", None)
            session = self.get_session(session_id, user)
            if not session:
                return self.json_error_response("Chat session not found", status=404)

            try:
                message = session.messages.get(id=message_id)
                message.soft_delete()

                return self.json_response(
                    {
                        "message": "Chat message deleted successfully",
                        "message_id": message_id,
                        "is_deleted": message.is_deleted,
                    }
                )

            except ChatHistory.DoesNotExist:
                return self.json_error_response("Message not found", status=404)

        except Exception as e:
            return self.json_error_response(str(e), status=500)

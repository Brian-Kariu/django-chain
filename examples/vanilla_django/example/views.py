"""
Comprehensive example views demonstrating all major django-chain features.

This module showcases:
- Prompt creation and management
- Workflow authoring and execution
- Chat session management
- Vector store operations
- InteractionLog usage
- Error handling patterns
- Integration with Django authentication
"""

import json
import uuid
from typing import Any
from typing import Dict
from typing import List

from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import ValidationError
from django.http import HttpRequest
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django_chain.config import app_settings
from django_chain.exceptions import LLMProviderAPIError
from django_chain.exceptions import PromptValidationError
from django_chain.exceptions import WorkflowValidationError
from django_chain.models import ChatHistory
from django_chain.models import ChatSession
from django_chain.models import InteractionLog
from django_chain.models import Prompt
from django_chain.models import Workflow
from django_chain.services.vector_store_manager import VectorStoreManager
from django_chain.utils.llm_client import create_llm_chat_client
from django_chain.utils.llm_client import create_llm_embedding_client
from django_chain.utils.memory_manager import get_langchain_memory

from .models import ExampleDocument

# =============================================================================
# DASHBOARD & OVERVIEW
# =============================================================================


def dashboard(request: HttpRequest):
    """Main dashboard showing django-chain capabilities and examples."""
    context = {
        "total_prompts": Prompt.objects.count(),
        "total_workflows": Workflow.objects.count(),
        "total_chat_sessions": ChatSession.objects.count(),
        "total_interactions": InteractionLog.objects.count(),
        "app_settings": {
            "provider": app_settings.DEFAULT_LLM_PROVIDER,
            "model": app_settings.DEFAULT_CHAT_MODEL.get("name"),
            "logging_enabled": app_settings.ENABLE_LLM_LOGGING,
        },
    }
    return render(request, "example/dashboard.html", context)


def api_overview(request: HttpRequest):
    """API overview and documentation."""
    return JsonResponse(
        {
            "django_chain_features": {
                "prompts": "/django-chain/prompts/",
                "workflows": "/django-chain/workflows/",
                "chat_sessions": "/django-chain/chat/sessions/",
                "interactions": "/django-chain/interactions/",
                "execute_workflow": "/django-chain/workflows/{name}/execute/",
            },
            "example_endpoints": {
                "create_prompt": "/example/prompts/create/",
                "create_workflow": "/example/workflows/create/",
                "chat_demo": "/example/chat/",
                "vector_demo": "/example/vector/",
                "llm_test": "/example/llm-test/",
            },
        }
    )


# =============================================================================
# PROMPT MANAGEMENT EXAMPLES
# =============================================================================


@csrf_exempt
@require_http_methods(["GET", "POST"])
def prompt_examples(request: HttpRequest):
    """Demonstrate prompt creation and management."""
    if request.method == "GET":
        prompts = Prompt.objects.all().order_by("-created_at")[:10]
        return render(request, "example/prompts.html", {"prompts": prompts})

    try:
        data = json.loads(request.body)

        # Example: Create a simple prompt template
        if data.get("example_type") == "simple":
            prompt = Prompt.objects.create(
                name=f"example_simple_{uuid.uuid4().hex[:8]}",
                prompt_template={
                    "langchain_type": "PromptTemplate",
                    "template": "You are a helpful assistant. User question: {question}",
                    "input_variables": ["question"],
                },
                input_variables=["question"],
                is_active=True,
            )

        # Example: Create a chat prompt template
        elif data.get("example_type") == "chat":
            prompt = Prompt.objects.create(
                name=f"example_chat_{uuid.uuid4().hex[:8]}",
                prompt_template={
                    "langchain_type": "ChatPromptTemplate",
                    "messages": [
                        {"type": "system", "content": "You are a {role} assistant."},
                        {"type": "human", "content": "{user_input}"},
                    ],
                },
                input_variables=["role", "user_input"],
                is_active=True,
            )

        # Example: Create a custom prompt
        else:
            prompt_data = data.get("prompt_data", {})
            prompt = Prompt.objects.create(
                name=prompt_data.get("name", f"custom_{uuid.uuid4().hex[:8]}"),
                prompt_template=prompt_data.get("template", {}),
                input_variables=prompt_data.get("input_variables", []),
                optional_variables=prompt_data.get("optional_variables", []),
                is_active=True,
            )

        return JsonResponse(
            {
                "status": "success",
                "prompt_id": str(prompt.id),
                "name": prompt.name,
                "langchain_prompt": str(prompt.to_langchain_prompt()),
            }
        )

    except (ValidationError, PromptValidationError) as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)
    except Exception as e:
        return JsonResponse(
            {"status": "error", "message": f"Unexpected error: {str(e)}"}, status=500
        )


# =============================================================================
# WORKFLOW EXAMPLES
# =============================================================================


@csrf_exempt
@require_http_methods(["GET", "POST"])
def workflow_examples(request: HttpRequest):
    """Demonstrate workflow creation and execution."""
    if request.method == "GET":
        workflows = Workflow.objects.all().order_by("-created_at")[:10]
        return render(request, "example/workflows.html", {"workflows": workflows})

    try:
        data = json.loads(request.body)

        # Create a prompt first if needed
        prompt_name = f"workflow_prompt_{uuid.uuid4().hex[:8]}"
        prompt = Prompt.objects.create(
            name=prompt_name,
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": data.get("prompt_template", "Answer this question: {question}"),
                "input_variables": ["question"],
            },
            input_variables=["question"],
            is_active=True,
        )

        # Create workflow with different complexity levels
        workflow_type = data.get("workflow_type", "simple")

        if workflow_type == "simple":
            # Simple: Prompt -> LLM
            workflow_definition = [
                {"type": "prompt", "name": prompt_name},
                {"type": "llm", "provider": app_settings.DEFAULT_LLM_PROVIDER},
            ]
        elif workflow_type == "with_parser":
            # With parser: Prompt -> LLM -> Parser
            workflow_definition = [
                {"type": "prompt", "name": prompt_name},
                {"type": "llm", "provider": app_settings.DEFAULT_LLM_PROVIDER},
                {"type": "parser", "parser_type": "str"},
            ]
        else:
            # Custom workflow
            workflow_definition = data.get("workflow_definition", [])

        workflow = Workflow.objects.create(
            name=f"example_workflow_{uuid.uuid4().hex[:8]}",
            description=data.get("description", "Example workflow created via API"),
            prompt=prompt,
            workflow_definition=workflow_definition,
            is_active=True,
        )

        return JsonResponse(
            {
                "status": "success",
                "workflow_id": str(workflow.id),
                "name": workflow.name,
                "definition": workflow.workflow_definition,
            }
        )

    except (ValidationError, WorkflowValidationError) as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)
    except Exception as e:
        return JsonResponse(
            {"status": "error", "message": f"Unexpected error: {str(e)}"}, status=500
        )


@csrf_exempt
@require_http_methods(["POST"])
def execute_workflow_example(request: HttpRequest):
    """Demonstrate workflow execution with logging and error handling."""
    try:
        data = json.loads(request.body)
        workflow_name = data.get("workflow_name")
        input_data = data.get("input", {})
        session_id = data.get("session_id")

        if not workflow_name:
            return JsonResponse(
                {"status": "error", "message": "workflow_name required"}, status=400
            )

        workflow = get_object_or_404(Workflow, name=workflow_name, is_active=True)

        # Execute with logging enabled
        chain = workflow.to_langchain_chain(
            log="true",
            llm_config={
                "DEFAULT_LLM_PROVIDER": app_settings.DEFAULT_LLM_PROVIDER,
                "DEFAULT_CHAT_MODEL": app_settings.DEFAULT_CHAT_MODEL,
            },
            user=request.user if request.user.is_authenticated else None,
            session_id=session_id,
        )

        # Execute the workflow
        result = chain.invoke(input_data)

        # Get the latest interaction log for this execution
        latest_log = (
            InteractionLog.objects.filter(workflow=workflow).order_by("-created_at").first()
        )

        return JsonResponse(
            {
                "status": "success",
                "result": str(result),
                "workflow_name": workflow_name,
                "session_id": session_id,
                "interaction_log_id": str(latest_log.id) if latest_log else None,
                "tokens_used": {
                    "input": latest_log.input_tokens if latest_log else None,
                    "output": latest_log.output_tokens if latest_log else None,
                },
            }
        )

    except Workflow.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Workflow not found"}, status=404)
    except Exception as e:
        return JsonResponse(
            {"status": "error", "message": f"Execution failed: {str(e)}"}, status=500
        )


# =============================================================================
# CHAT SESSION EXAMPLES
# =============================================================================


class ChatDemoView(View):
    """Comprehensive chat demonstration with session management."""

    def get(self, request: HttpRequest):
        """Show chat interface and recent sessions."""
        recent_sessions = ChatSession.objects.all().order_by("-created_at")[:5]
        return render(request, "example/chat.html", {"recent_sessions": recent_sessions})

    @method_decorator(csrf_exempt)
    def post(self, request: HttpRequest):
        """Handle chat messages with session management."""
        try:
            data = json.loads(request.body)
            message = data.get("message")
            session_id = data.get("session_id")

            if not message:
                return JsonResponse({"status": "error", "message": "Message required"}, status=400)

            # Get or create chat session
            if session_id:
                # Convert session_id to UUID if it's a string
                if isinstance(session_id, str):
                    try:
                        session_id_uuid = uuid.UUID(session_id)
                    except ValueError:
                        return JsonResponse(
                            {
                                "status": "error",
                                "message": f"Invalid session_id format: {session_id}. Must be a valid UUID.",
                            },
                            status=400,
                        )
                else:
                    session_id_uuid = session_id

                try:
                    session = ChatSession.objects.get(session_id=session_id_uuid)
                except ChatSession.DoesNotExist:
                    session = self._create_chat_session(session_id_uuid, request.user)
            else:
                session_id_uuid = uuid.uuid4()
                session = self._create_chat_session(session_id_uuid, request.user)
                session_id = str(session_id_uuid)  # For response

            # Log user message
            user_message = ChatHistory.objects.create(
                session=session, content=message, role="USER", order=session.messages.count()
            )

            # Get LLM response
            provider = app_settings.DEFAULT_LLM_PROVIDER
            llm = create_llm_chat_client(provider)

            if llm is None:
                raise LLMProviderAPIError(f"Failed to initialize {provider} client")

            response = llm.invoke(message)
            response_content = response.content if hasattr(response, "content") else str(response)

            # Log assistant message
            assistant_message = ChatHistory.objects.create(
                session=session,
                content=response_content,
                role="ASSISTANT",
                order=session.messages.count(),
            )

            return JsonResponse(
                {
                    "status": "success",
                    "response": response_content,
                    "session_id": str(session.session_id),
                    "message_count": session.messages.count(),
                    "user_message_id": user_message.id,
                    "assistant_message_id": assistant_message.id,
                }
            )

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    def _create_chat_session(self, session_id_uuid: uuid.UUID, user):
        """Create a new chat session."""
        return ChatSession.objects.create(
            session_id=session_id_uuid,
            title=f"Chat Session {str(session_id_uuid)[:8]}",
            user=user if user.is_authenticated else None,
        )


# =============================================================================
# VECTOR STORE EXAMPLES
# =============================================================================


class VectorStoreDemo(View):
    """Demonstrate vector store operations and document management."""

    def get(self, request: HttpRequest):
        """Show vector store interface and recent documents."""
        documents = ExampleDocument.objects.all().order_by("-created_at")[:10]
        return render(request, "example/vector.html", {"documents": documents})

    @method_decorator(csrf_exempt)
    def post(self, request: HttpRequest):
        """Handle document operations (add, search)."""
        try:
            data = json.loads(request.body)
            operation = data.get("operation")

            if operation == "add_document":
                return self._add_document(data)
            elif operation == "search":
                return self._search_documents(data)
            else:
                return JsonResponse({"status": "error", "message": "Invalid operation"}, status=400)

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    def _add_document(self, data: Dict[str, Any]):
        """Add a document to the vector store."""
        content = data.get("content")
        title = data.get("title", "Untitled Document")

        if not content:
            return JsonResponse({"status": "error", "message": "Content required"}, status=400)

        # Save to database
        doc = ExampleDocument.objects.create(title=title, content=content)

        # Add to vector store
        try:
            manager = VectorStoreManager()
            manager.add_documents_to_store([content])

            return JsonResponse(
                {
                    "status": "success",
                    "document_id": doc.id,
                    "title": doc.title,
                    "message": "Document added to vector store",
                }
            )
        except Exception as e:
            # Clean up database record if vector store fails
            doc.delete()
            raise e

    def _search_documents(self, data: Dict[str, Any]):
        """Search documents in the vector store."""
        query = data.get("query")
        k = data.get("k", 5)

        if not query:
            return JsonResponse({"status": "error", "message": "Query required"}, status=400)

        manager = VectorStoreManager()
        results = manager.search_documents(query, k=k)

        return JsonResponse(
            {"status": "success", "query": query, "results": results, "count": len(results)}
        )


# =============================================================================
# LLM TESTING & UTILITIES
# =============================================================================


@csrf_exempt
@require_http_methods(["GET", "POST"])
def llm_test_view(request: HttpRequest):
    """Test LLM providers and configurations."""
    if request.method == "GET":
        return render(
            request,
            "example/llm_test.html",
            {
                "providers": ["openai", "google", "huggingface", "fake"],
                "current_provider": app_settings.DEFAULT_LLM_PROVIDER,
                "current_model": app_settings.DEFAULT_CHAT_MODEL.get("name"),
            },
        )

    try:
        data = json.loads(request.body)
        test_type = data.get("test_type", "chat")
        provider = data.get("provider", app_settings.DEFAULT_LLM_PROVIDER)
        message = data.get("message", "Hello, how are you?")

        if test_type == "chat":
            return _test_chat_model(provider, message)
        elif test_type == "embedding":
            return _test_embedding_model(provider, message)
        else:
            return JsonResponse({"status": "error", "message": "Invalid test_type"}, status=400)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def _test_chat_model(provider: str, message: str):
    """Test chat model functionality."""
    try:
        llm = create_llm_chat_client(provider)
        if llm is None:
            return JsonResponse(
                {"status": "error", "message": f"Failed to create {provider} client"}, status=500
            )

        response = llm.invoke(message)
        response_content = response.content if hasattr(response, "content") else str(response)

        return JsonResponse(
            {
                "status": "success",
                "provider": provider,
                "message": message,
                "response": response_content,
                "model_info": {
                    "class": response.__class__.__name__
                    if hasattr(response, "__class__")
                    else "Unknown",
                    "has_content": hasattr(response, "content"),
                    "has_usage": hasattr(response, "usage_metadata"),
                },
            }
        )
    except Exception as e:
        return JsonResponse(
            {"status": "error", "message": f"Chat test failed: {str(e)}"}, status=500
        )


def _test_embedding_model(provider: str, text: str):
    """Test embedding model functionality."""
    try:
        embedding_model = create_llm_embedding_client(provider)
        if embedding_model is None:
            return JsonResponse(
                {"status": "error", "message": f"Failed to create {provider} embedding client"},
                status=500,
            )

        embeddings = embedding_model.embed_documents([text])

        return JsonResponse(
            {
                "status": "success",
                "provider": provider,
                "text": text,
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
                "embedding_preview": embeddings[0][:5] if embeddings else [],
            }
        )
    except Exception as e:
        return JsonResponse(
            {"status": "error", "message": f"Embedding test failed: {str(e)}"}, status=500
        )


# =============================================================================
# INTERACTION LOG EXAMPLES
# =============================================================================


def interaction_logs_view(request: HttpRequest):
    """Display recent interaction logs with filtering."""
    logs = InteractionLog.objects.all().order_by("-created_at")[:20]

    # Group by status for summary
    summary = {
        "total": logs.count(),
        "success": logs.filter(status="SUCCESS").count(),
        "failure": logs.filter(status="FAILURE").count(),
        "processing": logs.filter(status="PROCESSING").count(),
    }

    return render(request, "example/interaction_logs.html", {"logs": logs, "summary": summary})


# =============================================================================
# ERROR HANDLING EXAMPLES
# =============================================================================


@csrf_exempt
def error_handling_demo(request: HttpRequest):
    """Demonstrate various error handling scenarios."""
    try:
        data = json.loads(request.body)
        error_type = data.get("error_type")

        if error_type == "prompt_validation":
            # Trigger prompt validation error
            Prompt.objects.create(
                name="invalid_prompt",
                prompt_template={"invalid": "template"},  # Missing langchain_type
                is_active=True,
            )
        elif error_type == "workflow_validation":
            # Trigger workflow validation error
            Workflow.objects.create(
                name="invalid_workflow",
                workflow_definition=[{"invalid": "step"}],  # Invalid step format
                is_active=True,
            )
        elif error_type == "llm_provider":
            # Trigger LLM provider error
            llm = create_llm_chat_client("nonexistent_provider")
            if llm:
                llm.invoke("test")
        else:
            # Generic error
            raise Exception("Demo error for testing")

        return JsonResponse({"status": "error", "message": "Should not reach here"})

    except (PromptValidationError, WorkflowValidationError) as e:
        return JsonResponse(
            {"status": "validation_error", "error_type": type(e).__name__, "message": str(e)},
            status=400,
        )
    except LLMProviderAPIError as e:
        return JsonResponse(
            {"status": "llm_error", "error_type": type(e).__name__, "message": str(e)}, status=503
        )
    except Exception as e:
        return JsonResponse(
            {"status": "unexpected_error", "error_type": type(e).__name__, "message": str(e)},
            status=500,
        )


# =============================================================================
# UTILITY FUNCTIONS FOR CUSTOM WORKFLOWS
# =============================================================================


def custom_workflow_builder_demo(request: HttpRequest):
    """Demonstrate how developers can build custom workflows using utility functions."""
    if request.method == "GET":
        return render(request, "example/custom_workflow.html")

    try:
        # Example of building a custom workflow programmatically
        from django_chain.utils.langchain import Workflow as WorkflowProcessor

        # Step 1: Create a custom prompt
        custom_prompt = Prompt.objects.create(
            name=f"custom_builder_{uuid.uuid4().hex[:8]}",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": "Analyze this text and provide insights: {text}",
                "input_variables": ["text"],
            },
            input_variables=["text"],
            is_active=True,
        )

        # Step 2: Define workflow steps
        workflow_steps = [
            {"type": "prompt", "name": custom_prompt.name},
            {"type": "llm", "provider": "fake"},
            {"type": "parser", "parser_type": "str"},
        ]

        # Step 3: Create and save workflow
        custom_workflow = Workflow.objects.create(
            name=f"custom_analysis_{uuid.uuid4().hex[:8]}",
            description="Custom text analysis workflow built programmatically",
            prompt=custom_prompt,
            workflow_definition=workflow_steps,
            is_active=True,
        )

        # Step 4: Execute the workflow
        chain = custom_workflow.to_langchain_chain(
            log="true",
            llm_config={
                "DEFAULT_LLM_PROVIDER": app_settings.DEFAULT_LLM_PROVIDER,
                "DEFAULT_CHAT_MODEL": app_settings.DEFAULT_CHAT_MODEL,
            },
        )

        result = chain.invoke({"text": "This is a sample text for analysis."})

        return JsonResponse(
            {
                "status": "success",
                "workflow_id": str(custom_workflow.id),
                "workflow_name": custom_workflow.name,
                "result": str(result),
                "steps_executed": len(workflow_steps),
            }
        )

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

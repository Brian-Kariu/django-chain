"""
URL configuration for django-chain vanilla Django example project.

This project demonstrates all major django-chain features:
- Prompt management and creation
- Workflow authoring and execution
- Chat session management with history
- Vector store operations and document management
- LLM provider testing and configuration
- Interaction logging and analytics
- Error handling patterns
- Custom workflow building utilities
"""

from django.contrib import admin
from django.http import JsonResponse
from django.urls import include
from django.urls import path


def project_info(request):
    """Project information and navigation."""
    return JsonResponse(
        {
            "project": "Django-Chain Vanilla Example",
            "description": "Comprehensive demonstration of django-chain features",
            "features_demonstrated": [
                "Prompt Management",
                "Workflow Creation & Execution",
                "Chat Session Management",
                "Vector Store Operations",
                "LLM Provider Testing",
                "Interaction Logging",
                "Error Handling",
                "Custom Workflow Building",
            ],
            "endpoints": {
                "admin": "/admin/",
                "examples": "/example/",
                "django_chain_api": "/django-chain/",
                "api_docs": "/example/api/",
            },
            "quick_start": {
                "dashboard": "/example/",
                "create_prompt": "/example/prompts/",
                "create_workflow": "/example/workflows/",
                "test_chat": "/example/chat/",
                "test_llm": "/example/llm-test/",
            },
        }
    )


urlpatterns = [
    # Project info
    path("", project_info, name="project_info"),
    # Django admin
    path("admin/", admin.site.urls),
    # Example application demonstrating django-chain features
    path("example/", include("examples.vanilla_django.example.urls")),
    # Direct access to django-chain APIs
    path("django-chain/", include("django_chain.urls")),
]

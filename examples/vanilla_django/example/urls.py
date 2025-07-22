"""
URL configuration for comprehensive django-chain examples.

This module demonstrates all major features:
- Dashboard and API overview
- Prompt management examples
- Workflow creation and execution
- Chat session management
- Vector store operations
- LLM testing utilities
- Interaction logging
- Error handling patterns
- Custom workflow building
"""

from django.urls import path, include

from examples.vanilla_django.example import views

app_name = "example"

urlpatterns = [
    # Dashboard and overview
    path("", views.dashboard, name="dashboard"),
    path("api/", views.api_overview, name="api_overview"),
    # Prompt management examples
    path("prompts/", views.prompt_examples, name="prompt_examples"),
    # Workflow examples
    path("workflows/", views.workflow_examples, name="workflow_examples"),
    path("workflows/execute/", views.execute_workflow_example, name="execute_workflow"),
    # Chat functionality
    path("chat/", views.ChatDemoView.as_view(), name="chat_demo"),
    # Vector store operations
    path("vector/", views.VectorStoreDemo.as_view(), name="vector_demo"),
    # LLM testing
    path("llm-test/", views.llm_test_view, name="llm_test"),
    # Interaction logs
    path("logs/", views.interaction_logs_view, name="interaction_logs"),
    # Error handling examples
    path("errors/", views.error_handling_demo, name="error_demo"),
    # Custom workflow builder
    path("custom-workflow/", views.custom_workflow_builder_demo, name="custom_workflow"),
    # Include django-chain URLs for full API access
    path("django-chain/", include("django_chain.urls")),
]

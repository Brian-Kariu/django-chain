from django.urls import path

from django_chain import views

urlpatterns = [
    path("prompts/", views.PromptListCreateView.as_view(), name="prompt-list-create"),
    path("prompts/<uuid:pk>/", views.PromptDetailView.as_view(), name="prompt-detail"),
    path(
        "prompts/<uuid:pk>/activate/",
        views.PromptActivateDeactivateView.as_view(),
        {"action": "activate"},
        name="prompt-activate",
    ),
    path(
        "prompts/<uuid:pk>/deactivate/",
        views.PromptActivateDeactivateView.as_view(),
        {"action": "deactivate"},
        name="prompt-deactivate",
    ),
    path("workflows/", views.WorkflowListCreateView.as_view(), name="workflow-list-create"),
    path("workflows/<uuid:pk>/", views.WorkflowDetailView.as_view(), name="workflow-detail"),
    path(
        "workflows/<uuid:pk>/activate/",
        views.WorkflowActivateDeactivateView.as_view(),
        {"action": "activate"},
        name="workflow-activate",
    ),
    path(
        "workflows/<uuid:pk>/deactivate/",
        views.WorkflowActivateDeactivateView.as_view(),
        {"action": "deactivate"},
        name="workflow-deactivate",
    ),
    path(
        "workflows/execute/<str:name>/",
        views.ExecuteWorkflowView.as_view(),
        name="execute-workflow",
    ),
    path("logs/", views.InteractionLogListCreateView.as_view(), name="log-list-create"),
    path("logs/<uuid:pk>/", views.InteractionLogDetailView.as_view(), name="logs-detail"),
    path(
        "chat/sessions/", views.ChatSessionListCreateView.as_view(), name="chat-session-list-create"
    ),
    path(
        "chat/sessions/<str:session_id>/",
        views.ChatSessionDetailView.as_view(),
        name="chat-session-detail",
    ),
    path(
        "chat/sessions/<str:session_id>/messages/",
        views.ChatHistoryView.as_view(),
        name="chat-history",
    ),
    path(
        "chat/sessions/<str:session_id>/messages/<int:message_id>/",
        views.ChatHistoryView.as_view(),
        name="chat-history-delete",
    ),
    # Legacy endpoints (for backward compatibility)
    path("chat/", views.chat_view, name="chat-view"),
    path("vector-search/", views.vector_search_view, name="vector-search-view"),
]

"""
URL configuration for test app.
"""

from django.urls import path

from tests.test_project.testapp import views

app_name = "testapp"

urlpatterns = [
    path("test-llm/", views.test_llm_call, name="test_llm"),
    path("test-chain/", views.test_chain_execution, name="test_chain"),
    path("test-chat/", views.test_chat_session, name="test_chat"),
    path("test-vector-store/", views.test_vector_store, name="test_vector_store"),
    path("test-prompt/", views.TestAppPrompts.as_view(), name="test_prompt"),
    path("test-prompt/<str:prompt_id>/", views.TestAppPrompts.as_view(), name="test_prompt_by_id"),
]

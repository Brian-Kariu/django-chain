"""
URL configuration for test app.
"""

from django.urls import path

from . import views

app_name = "testapp"

urlpatterns = [
    path("test-llm/", views.test_llm_call, name="test_llm"),
    path("test-chain/", views.test_chain_execution, name="test_chain"),
    path("test-chat/", views.test_chat_session, name="test_chat"),
    path("test-vector-store/", views.test_vector_store, name="test_vector_store"),
]

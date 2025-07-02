"""
URL configuration for test app.
"""

from django.urls import include
from django.urls import path

from examples.vanilla_django.example import views

app_name = "testapp"

urlpatterns = [
    path("test-llm/", views.test_llm_call, name="test_llm"),
    path("test-chat/", views.test_chat_session, name="test_chat"),
    path("django-chain/", include("django_chain.urls")),
]

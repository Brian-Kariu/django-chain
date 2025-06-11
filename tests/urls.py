from typing import Any

from django.urls import path

from django_chain.views import LLMView

from .test_project.testapp import views


class TestLLMView(LLMView):
    def get_prompt_context(self, request):
        return {"input": request.GET.get("input", "")}


urlpatterns: list[Any] = [
    path("test/", TestLLMView.as_view(), name="test_llm"),
    path("chat/", views.test_chat_view, name="chat"),
    path("vector-search/", views.test_vector_search_view, name="vector_search"),
    path("document/", views.test_document_view, name="document"),
    path("document/list/", views.test_document_list_view, name="document_list"),
]

# myproject/myapp/urls.py
from django.urls import path

from django_chain.views import PromptView

urlpatterns = [
    path("prompt/<int:prompt_id>/", PromptView.as_view(), name="prompt_view"),
]

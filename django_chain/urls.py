# myproject/myapp/urls.py
from django.urls import path

from django_chain import views

urlpatterns = [
    path("prompts/", views.prompt_list_create, name="prompt-list-create"),
    path("prompts/<uuid:pk>/", views.prompt_detail, name="prompt-detail"),
    path("prompts/<uuid:pk>/activate/", views.prompt_activate, name="prompt-activate"),
    path("prompts/<uuid:pk>/deactivate/", views.prompt_deactivate, name="prompt-deactivate"),
    path(
        "prompts/active/<str:name>/",
        views.prompt_list_create,
        {"include_inactive": False},
        name="prompt-active-by-name",
    ),
]

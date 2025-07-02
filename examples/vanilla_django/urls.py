"""
URL configuration for test project.
"""

from django.contrib import admin
from django.urls import include
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("django-chain/", include("django_chain.urls")),
]

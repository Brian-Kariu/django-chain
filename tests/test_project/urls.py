"""
URL configuration for test project.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("test/", include("test_project.testapp.urls", namespace="testapp")),
]

"""
Test application configuration.
"""

from django.apps import AppConfig


class TestAppConfig(AppConfig):
    """
    Configuration for the test application.
    """

    name = "examples.vanilla_django.example"
    verbose_name = "Test Application"

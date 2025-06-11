"""
Test application configuration.
"""

from django.apps import AppConfig


class TestAppConfig(AppConfig):
    """
    Configuration for the test application.
    """

    name = "tests.test_project.testapp"
    label = "testapp"
    verbose_name = "Test Application"

    def ready(self):
        """
        Perform any initialization when the app is ready.
        """
        try:
            import tests.test_project.testapp.signals  # noqa
        except ImportError:
            pass

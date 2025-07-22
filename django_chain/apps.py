from django.apps import AppConfig


class DjangoChainConfig(AppConfig):
    name = "django_chain"
    verbose_name = "Django Chain"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        """
        Perform initialization when the app is ready.
        This includes validating configuration settings.
        """
        try:
            from django_chain.config import validate_settings

            validate_settings()
        except Exception:
            # In development/testing, we might want to allow invalid settings
            # to be caught later when actually used, rather than preventing startup
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Django Chain settings validation failed. "
                "Some features may not work correctly. "
                "Please check your DJANGO_LLM_SETTINGS configuration."
            )

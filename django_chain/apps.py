from django.apps import AppConfig


class DjangoChainConfig(AppConfig):
    name = "django_chain"
    verbose_name = "Django Chain"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        try:
            import django_chain.signals
        except ImportError:
            pass

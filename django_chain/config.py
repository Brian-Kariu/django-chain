from copy import deepcopy

from django.conf import settings


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    defaults = {
        "DEFAULT_LLM_PROVIDER": "fake",
        "DEFAULT_CHAT_MODEL": {
            "name": "fake-model",
            "api_key": "FAKE_API_KEY",
            "temperature": 0.5,
            "max_tokens": 1024,
        },
        "DEFAULT_EMBEDDING_MODEL": {
            "provider": "fake",
            "name": "fake-embedding",
        },
        "VECTOR_STORE": {
            "TYPE": "",
            "PGVECTOR_COLLECTION_NAME": "",
        },
        "ENABLE_LLM_LOGGING": False,
        "LLM_LOGGING_LEVEL": "DEBUG",
        "MEMORY": {"STORE": "INMEMORY"},  # TODO: Add other connection related configs
        "CHAIN": {
            "DEFAULT_OUTPUT_PARSER": "str",
            "ENABLE_MEMORY": False,
        },
        "CACHING": {
            "CACHE_LLM_RESPONSES": False,
            "CACHE_TTL_SECONDS": 3600,
        },
    }

    def _setup(self):
        app_settings = getattr(settings, "DJANGO_LLM_SETTINGS", {})
        self.attrs = deepcopy(self.defaults)

        self.attrs.update(app_settings)

    def __init__(self):
        super().__init__()
        self._setup()

    def __getattr__(self, item):
        return self.attrs.get(item, None)

    def __setattribute__(self, key, value):
        self.attrs[key] = value

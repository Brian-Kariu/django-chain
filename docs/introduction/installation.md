# Installation

To install Django Chain, follow these steps:

```bash
pip install django-chain
```

Add `django_chain` to your `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    ...
    'django_chain',
    ...
]
```

Add you LLM model configurations:

```python
DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "fake",
    "DEFAULT_CHAT_MODEL": {
        "name": "fake-model",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": "fake key",
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "fake",
        "name": "fake-embedding",
    },
    "VECTOR_STORE": {
        "TYPE": "pgvector",
        "PGVECTOR_COLLECTION_NAME": "test_documents",
    },
    "ENABLE_LLM_LOGGING": True,
    "LLM_LOGGING_LEVEL": "DEBUG",
    "MEMORY": {
        "DEFAULT_TYPE": "buffer",
        "WINDOW_SIZE": 5,
    },
    "CHAIN": {
        "DEFAULT_OUTPUT_PARSER": "str",
        "ENABLE_MEMORY": True,
    },
    "CACHE_LLM_RESPONSES": True,
    "CACHE_TTL_SECONDS": 3600,
}
```

Run migrations:
```bash
python manage.py makemigrations django_chain
python manage.py migrate django_chain
```

## Quick Start
Add these urls to your app:
```python
# your_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('api/', include('django_chain.urls')), # Or your chosen app name
]
```

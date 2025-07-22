# Configuration Guide

Django Chain uses the Django App Settings pattern (similar to Django REST Framework) for configuration management. This provides a robust, testable, and extensible way to configure your LLM integration.

## 🏗️ **Configuration Architecture**

Django Chain loads settings from `settings.DJANGO_LLM_SETTINGS` with sensible defaults. The configuration system:

- **Validates settings** on Django app startup
- **Provides defaults** for all optional settings
- **Supports environment variables** for sensitive data
- **Warns about deprecated settings** during upgrades
- **Fails fast** with clear error messages for invalid configurations

## ⚙️ **Core Settings**

### **LLM Provider Configuration**

```python
DJANGO_LLM_SETTINGS = {
    # Primary LLM provider
    "DEFAULT_LLM_PROVIDER": "openai",  # "openai", "google", "huggingface", "fake"

    # Chat model configuration
    "DEFAULT_CHAT_MODEL": {
        "name": "gpt-3.5-turbo",
        "temperature": 0.7,        # Randomness (0.0 = deterministic, 1.0 = very random)
        "max_tokens": 1024,        # Maximum response length
        "top_p": 1.0,             # Nucleus sampling parameter
        "frequency_penalty": 0.0,  # Penalize repeated tokens
        "presence_penalty": 0.0,   # Penalize new topics
        "api_key": "your-api-key", # Use environment variables!
    },

    # Embedding model for vector operations
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "openai",
        "name": "text-embedding-ada-002",
        "api_key": "your-api-key",
    },
}
```

### **Vector Store Configuration**

```python
DJANGO_LLM_SETTINGS = {
    "VECTOR_STORE": {
        "TYPE": "pgvector",                    # "pgvector", "chroma", "pinecone"
        "PGVECTOR_COLLECTION_NAME": "documents", # Collection/table name

        # pgvector specific settings
        "PGVECTOR_VECTOR_SIZE": 1536,         # Embedding dimension
        "PGVECTOR_DISTANCE_STRATEGY": "cosine", # "cosine", "euclidean", "inner_product"

        # Connection settings (uses Django DATABASES by default)
        "CONNECTION_STRING": None,             # Optional override
    },
}
```

### **Memory Management**

```python
DJANGO_LLM_SETTINGS = {
    "MEMORY": {
        "PROVIDER": "django",         # "django", "inmemory"
        "DEFAULT_TYPE": "buffer",     # "buffer", "buffer_window", "summary"
        "WINDOW_SIZE": 10,            # For buffer_window memory
    },
}
```

### **Logging and Monitoring**

```python
DJANGO_LLM_SETTINGS = {
    # Enable comprehensive LLM interaction logging
    "ENABLE_LLM_LOGGING": True,
    "LLM_LOGGING_LEVEL": "INFO",       # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    # Sensitive data handling
    "LOG_SENSITIVE_DATA": False,       # Never log API keys, tokens
    "SANITIZE_LOGS": True,            # Sanitize potentially sensitive content

    # Performance monitoring
    "TRACK_TOKEN_USAGE": True,        # Track input/output tokens
    "TRACK_LATENCY": True,           # Track response times
    "TRACK_ERRORS": True,            # Track and categorize errors
}
```

### **Chain and Workflow Configuration**

```python
DJANGO_LLM_SETTINGS = {
    "CHAIN": {
        "DEFAULT_OUTPUT_PARSER": "str",    # "str", "json", "xml", "yaml"
        "ENABLE_MEMORY": True,             # Include conversation memory
        "MAX_RETRIES": 3,                  # Retry failed LLM calls
        "RETRY_DELAY": 1.0,               # Seconds between retries
        "TIMEOUT": 30.0,                  # Request timeout in seconds
    },
}
```

### **Caching Configuration**

```python
DJANGO_LLM_SETTINGS = {
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": True,      # Cache LLM responses
        "CACHE_TTL_SECONDS": 3600,        # Cache lifetime (1 hour)
        "CACHE_KEY_PREFIX": "django_chain:", # Redis/cache key prefix
        "CACHE_BACKEND": "default",        # Django cache backend to use

        # Cache invalidation
        "INVALIDATE_ON_MODEL_CHANGE": True, # Clear cache when prompts/workflows change
    },
}
```

## 🔐 **Security and Environment Variables**

### **Using Environment Variables**

**Never hardcode API keys** in your settings files. Use environment variables:

```python
# settings.py
import os

DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": os.getenv('LLM_PROVIDER', 'fake'),
    "DEFAULT_CHAT_MODEL": {
        "name": os.getenv('LLM_MODEL_NAME', 'gpt-3.5-turbo'),
        "temperature": os.getenv('LLM_TEMPERATURE', 0.7),
        "max_tokens": os.getenv('LLM_MAX_TOKENS', 1024),
        "api_key": os.getenv('OPENAI_API_KEY'),  # Required, no default
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": os.getenv('EMBEDDING_PROVIDER', 'openai'),
        "name": os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002'),
        "api_key": os.getenv('OPENAI_API_KEY'),
    },
}
```

### **Environment File Example**

Create a `.env` file in your project root:

```bash
# .env

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# API Keys
OPENAI_API_KEY=sk-your-actual-openai-api-key
GOOGLE_API_KEY=your-google-ai-api-key
HUGGINGFACE_API_KEY=your-huggingface-token

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/your_db

# Optional: Override defaults
DJANGO_DEBUG=True
DJANGO_SECRET_KEY=your-secret-key
```

## 🌍 **Environment-Specific Configuration**

### **Development Settings**

```python
# settings/development.py
from .base import *

DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "fake",  # No API costs during development
    "DEFAULT_CHAT_MODEL": {
        "name": "fake-model",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": "fake-key",
    },
    "ENABLE_LLM_LOGGING": True,
    "LLM_LOGGING_LEVEL": "DEBUG",    # Verbose logging for development
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": False, # Disable caching for testing
    },
}
```

### **Production Settings**

```python
# settings/production.py
from .base import *

DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": os.getenv('LLM_PROVIDER'),
    "DEFAULT_CHAT_MODEL": {
        "name": os.getenv('LLM_MODEL_NAME'),
        "temperature": os.getenv('LLM_TEMPERATURE'),
        "max_tokens": os.getenv('LLM_MAX_TOKENS'),
        "api_key": os.getenv('OPENAI_API_KEY'),
    },
    "ENABLE_LLM_LOGGING": True,
    "LLM_LOGGING_LEVEL": "INFO",     # Less verbose in production
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": True,  # Enable caching for performance
        "CACHE_TTL_SECONDS": 7200,    # 2 hour cache
    },
    # Production-specific security settings
    "LOG_SENSITIVE_DATA": False,
    "SANITIZE_LOGS": True,
}
```

### **Testing Settings**

```python
# settings/testing.py
from .base import *

DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "fake",  # Always use fake provider in tests
    "DEFAULT_CHAT_MODEL": {
        "name": "fake-model",
        "api_key": "fake-key",
    },
    "ENABLE_LLM_LOGGING": False,     # Disable logging for faster tests
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": False, # No caching in tests
    },
}
```

## 🔧 **Advanced Configuration**

### **Custom Provider Configuration**

```python
DJANGO_LLM_SETTINGS = {
    # Multiple provider configurations
    "PROVIDERS": {
        "openai": {
            "api_key": os.getenv('OPENAI_API_KEY'),
            "base_url": "https://api.openai.com/v1",  # Custom endpoint
            "timeout": 30.0,
        },
        "azure_openai": {
            "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
            "base_url": os.getenv('AZURE_OPENAI_ENDPOINT'),
            "api_version": "2023-12-01-preview",
        },
        "google": {
            "api_key": os.getenv('GOOGLE_API_KEY'),
            "region": "us-central1",
        },
    },
}
```

### **Workflow-Specific Settings**

```python
DJANGO_LLM_SETTINGS = {
    "WORKFLOW_SETTINGS": {
        "customer_support": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,  # More deterministic for support
            "max_tokens": 512,
        },
        "content_generation": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.8,  # More creative for content
            "max_tokens": 2048,
        },
    },
}
```

## 📊 **Configuration Validation**

Django Chain validates your configuration on startup and provides helpful error messages:

### **Common Validation Errors**

```python
# Invalid provider
"DEFAULT_LLM_PROVIDER": "invalid_provider"
# Error: DEFAULT_LLM_PROVIDER must be one of: openai, google, huggingface, fake

# Missing required settings
"DEFAULT_CHAT_MODEL": {"name": "gpt-3.5-turbo"}  # Missing api_key
# Error: api_key is required for DEFAULT_CHAT_MODEL

# Invalid temperature
"DEFAULT_CHAT_MODEL": {"temperature": 2.0}  # Must be 0.0-1.0
# Error: temperature must be between 0.0 and 1.0
```

### **Runtime Configuration Access**

Access configuration in your code using the app settings:

```python
from django_chain.config import app_settings

# Get current provider
provider = app_settings.DEFAULT_LLM_PROVIDER

# Get chat model configuration
chat_config = app_settings.DEFAULT_CHAT_MODEL

# Get specific setting with fallback
cache_enabled = app_settings.get_setting('CACHE_LLM_SETTINGS', {}).get('CACHE_LLM_RESPONSES', False)

# Get provider-specific API key
api_key = app_settings.get_provider_api_key('openai')
```

## 🔄 **Configuration Updates**

### **Runtime Configuration Updates**

```python
from django_chain.config import app_settings

# Update settings at runtime (useful for testing)
app_settings.update({
    'DEFAULT_LLM_PROVIDER': 'fake',
    'ENABLE_LLM_LOGGING': False,
})

# Reload configuration from Django settings
app_settings.reload()
```

### **Testing Configuration Overrides**

```python
from django.test import override_settings
from django_chain.config import app_settings

@override_settings(DJANGO_LLM_SETTINGS={
    'DEFAULT_LLM_PROVIDER': 'fake',
    'ENABLE_LLM_LOGGING': False,
})
def test_with_custom_config():
    app_settings.reload()  # Reload after override
    assert app_settings.DEFAULT_LLM_PROVIDER == 'fake'
```

## 📋 **Complete Configuration Example**

Here's a complete production-ready configuration example:

```python
# settings/production.py
import os

DJANGO_LLM_SETTINGS = {
    # Core LLM Settings
    "DEFAULT_LLM_PROVIDER": os.getenv('LLM_PROVIDER', 'openai'),
    "DEFAULT_CHAT_MODEL": {
        "name": os.getenv('LLM_MODEL_NAME', 'gpt-3.5-turbo'),
        "temperature": os.getenv('LLM_TEMPERATURE', 0.7),
        "max_tokens": os.getenv('LLM_MAX_TOKENS', 1024),
        "api_key": os.getenv('OPENAI_API_KEY'),
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": os.getenv('EMBEDDING_PROVIDER', 'openai'),
        "name": os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002'),
        "api_key": os.getenv('OPENAI_API_KEY'),
    },

    # Vector Store
    "VECTOR_STORE": {
        "TYPE": os.getenv('VECTOR_STORE_TYPE', 'pgvector'),
        "PGVECTOR_COLLECTION_NAME": os.getenv('VECTOR_COLLECTION_NAME', 'documents'),
    },

    # Memory Management
    "MEMORY": {
        "PROVIDER": "django",         # "django", "inmemory"
        "DEFAULT_TYPE": "buffer",     # "buffer", "buffer_window", "summary"
        "WINDOW_SIZE": 10,            # For buffer_window memory
    },

    # Logging and Monitoring
    "ENABLE_LLM_LOGGING": os.getenv('ENABLE_LLM_LOGGING', True),
    "LLM_LOGGING_LEVEL": os.getenv('LLM_LOGGING_LEVEL', 'INFO'),
    "TRACK_TOKEN_USAGE": True,
    "TRACK_LATENCY": True,
    "LOG_SENSITIVE_DATA": False,
    "SANITIZE_LOGS": True,

    # Performance and Reliability
    "CHAIN": {
        "DEFAULT_OUTPUT_PARSER": "str",
        "ENABLE_MEMORY": True,
        "MAX_RETRIES": os.getenv('LLM_MAX_RETRIES', 3 ),
        "RETRY_DELAY": os.getenv('LLM_RETRY_DELAY', 1.0),
        "TIMEOUT": os.getenv('LLM_TIMEOUT', 30.0),
    },

    # Caching
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": os.getenv('CACHE_LLM_RESPONSES', True),
        "CACHE_TTL_SECONDS": os.getenv('CACHE_TTL_SECONDS', 3600),
        "CACHE_KEY_PREFIX": "django_chain:",
        "INVALIDATE_ON_MODEL_CHANGE": True,
    },
}
```

## 📚 **Next Steps**

- **[Tutorials](../usage/tutorials.md)**: Learn how to use these configurations in practice
- **[API Reference](../api/intro.md)**: Detailed documentation of configuration options
- **[Security Guide](../advanced/security.md)**: Security best practices for production deployments

# Installation Guide

This guide will walk you through installing and setting up Django Chain in your Django project.

## 📋 **Prerequisites**

- **Python**: 3.9+ (recommended: 3.11+)
- **Django**: 4.2+ or 5.0+
- **Database**: PostgreSQL (recommended for vector store features), SQLite (development), or MySQL

## 🚀 **Installation**

### **1. Install Django Chain**

```bash
pip install django-chain
```

For specific provider support, install optional dependencies:

```bash
# OpenAI support
pip install "django-chain[openai]"

# Google AI support
pip install "django-chain[google]"

# HuggingFace support
pip install "django-chain[huggingface]"

# Vector store support (pgvector)
pip install "django-chain[pgvector]"

# All optional dependencies
pip install "django-chain[all]"

# Development dependencies
pip install "django-chain[dev]"
```

### **2. Add to Django Settings**

Add `django_chain` to your `INSTALLED_APPS`:

```python
# settings.py
INSTALLED_APPS = [
    # ... your other apps
    'django_chain',
    # ... your other apps
]
```

### **3. Configure Django Chain Settings**

Add your LLM and Django Chain configuration to your Django settings:

```python
# settings.py
DJANGO_LLM_SETTINGS = {
    # LLM Provider Configuration
    "DEFAULT_LLM_PROVIDER": "openai",  # or "google", "huggingface", "fake"

    # Chat Model Settings
    "DEFAULT_CHAT_MODEL": {
        "name": "gpt-3.5-turbo",  # Model name
        "temperature": 0.7,       # Creativity level (0.0-1.0)
        "max_tokens": 1024,       # Maximum response length
        "api_key": "your-openai-api-key",  # API key (use environment variables!)
    },

    # Embedding Model Settings (for vector stores)
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "openai",
        "name": "text-embedding-ada-002",
        "api_key": "your-openai-api-key",  # Can be same as chat model
    },

    # Vector Store Configuration
    "VECTOR_STORE": {
        "TYPE": "pgvector",  # or "chroma", "pinecone"
        "PGVECTOR_COLLECTION_NAME": "django_chain_documents",
        # Additional connection parameters handled by Django's DATABASES setting
    },

    # Logging and Monitoring
    "ENABLE_LLM_LOGGING": True,
    "LLM_LOGGING_LEVEL": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Memory Management
    "MEMORY": {
        "PROVIDER": "django",         # "django", "inmemory"
        "DEFAULT_TYPE": "buffer",
        "WINDOW_SIZE": 5,
    },

    # Chain Configuration
    "CHAIN": {
        "DEFAULT_OUTPUT_PARSER": "str",  # "str", "json"
        "ENABLE_MEMORY": True,
    },

    # Caching (optional)
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": False,  # Set to True for development
        "CACHE_TTL_SECONDS": 3600,     # 1 hour
    },
}
```

### **4. Environment Variables (Recommended)**

For security, store API keys and sensitive settings in environment variables:

```python
# settings.py
import os

DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "openai",
    "DEFAULT_CHAT_MODEL": {
        "name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": os.getenv('OPENAI_API_KEY'),  # From environment
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "openai",
        "name": "text-embedding-ada-002",
        "api_key": os.getenv('OPENAI_API_KEY'),
    },
    # ... rest of your settings
}
```

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your-actual-openai-api-key
GOOGLE_API_KEY=your-google-ai-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

### **5. Database Setup**

Run Django migrations to create the necessary database tables:

```bash
# Create migration files (if needed)
python manage.py makemigrations django_chain

# Apply migrations
python manage.py migrate django_chain
```

### **6. URL Configuration**

Add Django Chain URLs to your project's URL configuration:

```python
# your_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Django Chain API endpoints
    path('api/llm/', include('django_chain.urls')),
    # ... your other URL patterns
]
```

This will make the following endpoints available:
- `/api/llm/prompts/` - Prompt management
- `/api/llm/workflows/` - Workflow management
- `/api/llm/chat/sessions/` - Chat session management
- `/api/llm/interactions/` - Interaction logs

## 🧪 **Verify Installation**

Create a simple test to verify everything is working:

```python
# test_django_chain.py
from django_chain.models import Prompt
from django_chain.config import app_settings

# Test configuration
print("Default LLM Provider:", app_settings.DEFAULT_LLM_PROVIDER)
print("Chat Model:", app_settings.DEFAULT_CHAT_MODEL)

# Test model creation
prompt = Prompt.objects.create(
    name="test_prompt",
    prompt_template={
        "langchain_type": "PromptTemplate",
        "template": "Hello {name}!",
        "input_variables": ["name"]
    },
    input_variables=["name"]
)
print(f"Created prompt: {prompt.name}")
```

## 🔧 **Provider-Specific Setup**

### **OpenAI Setup**

```python
DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "openai",
    "DEFAULT_CHAT_MODEL": {
        "name": "gpt-3.5-turbo",  # or "gpt-4", "gpt-4-turbo"
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": os.getenv('OPENAI_API_KEY'),
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "openai",
        "name": "text-embedding-ada-002",
        "api_key": os.getenv('OPENAI_API_KEY'),
    },
}
```

### **Google AI Setup**

```python
DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "google",
    "DEFAULT_CHAT_MODEL": {
        "name": "gemini-pro",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": os.getenv('GOOGLE_API_KEY'),
    },
}
```

### **HuggingFace Setup**

```python
DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "huggingface",
    "DEFAULT_CHAT_MODEL": {
        "name": "microsoft/DialoGPT-medium",
        "api_key": os.getenv('HUGGINGFACE_API_KEY'),  # Optional for some models
    },
}
```

### **Development/Testing Setup**

For development and testing without API costs:

```python
DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "fake",
    "DEFAULT_CHAT_MODEL": {
        "name": "fake-model",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": "fake-key",
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "fake",
        "name": "fake-embedding",
    },
}
```

## 🗄️ **Database Configuration for Vector Stores**

If you plan to use vector store features with pgvector:

### **PostgreSQL + pgvector Setup**

1. **Install pgvector extension** in your PostgreSQL database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. **Update Django database settings**:

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

3. **Configure vector store settings**:

```python
DJANGO_LLM_SETTINGS = {
    # ... other settings
    "VECTOR_STORE": {
        "TYPE": "pgvector",
        "PGVECTOR_COLLECTION_NAME": "my_documents",
    },
}
```

## 📝 **Next Steps**

Now that Django Chain is installed and configured:

1. **Explore the [Tutorials](../usage/tutorials.md)** for step-by-step guides
2. **Check out the [Vanilla Django Example](https://github.com/Brian-Kariu/django-chain/tree/main/examples/vanilla_django)** for a complete working project
3. **Read the [How-to Guides](../usage/how-to-guides.md)** for specific use cases
4. **Review the [API Reference](../api/intro.md)** for detailed documentation

## 🆘 **Troubleshooting**

### **Common Issues**

**Import Error**: If you get import errors, make sure `django_chain` is in your `INSTALLED_APPS` and you've run migrations.

**API Key Errors**: Ensure your API keys are correctly set in environment variables and accessible to your Django application.

**Database Errors**: For vector store features, make sure you have the pgvector extension installed in PostgreSQL.

**Configuration Errors**: Django Chain validates settings on startup. Check your Django logs for configuration validation errors.

### **Getting Help**

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/Brian-Kariu/django-chain/issues)
- **Documentation**: Check the [API Reference](../api/intro.md) for detailed information
- **Example Project**: See the [Vanilla Django Example](https://github.com/Brian-Kariu/django-chain/tree/main/examples/vanilla_django) for working code

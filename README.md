# Django Chain
A Django library for seamless LangChain integration, making it easy to add LLM capabilities to your Django applications.
This is a reusable Django application that provides a robust framework for defining, managing, and executing multi-step Large Language Model (LLM) workflows. It offers a set of API endpoints to interact with prompts, workflows, and their execution, enabling dynamic LLM applications without direct code changes for each new workflow.

![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FBrian-Kariu%2Fdjango-chain%2Fmain%2Fpyproject.toml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/Brian-Kariu/django-chain/graph/badge.svg?token=C2C53JBPKO)](https://codecov.io/gh/Brian-Kariu/django-chain)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Brian-Kariu/django-chain/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mkdocstrings.github.io/)

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/Brian-Kariu/django-chain/graph/badge.svg?token=C2C53JBPKO)](https://codecov.io/gh/Brian-Kariu/django-chain)
[![ci](https://github.com/Brian-Kariu/django-chain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Brian-Kariu/cookiecutter-airflow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

- Dynamic prompt and workflow management out of the box
- Easy integration with existing Django models and views
- Built-in utilities for common LLM tasks
- Type-safe and well-documented API
- Comprehensive test coverage
- Production-ready with proper error handling

## Core Concepts
- Prompt: Represents a configurable template for generating LLM prompts. This can be a HumanMessagePromptTemplate, SystemMessagePromptTemplate, or ChatPromptTemplate.
- Workflow: A sequence of ordered steps, each defining an action to be performed (e.g., format a prompt, call an LLM, parse JSON output, use a tool). Workflows orchestrate the flow of data through these steps.

## Prerequisites
1. Python 3.8+
2. Django 4.0+

## Installation

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

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
4. Run tests:
   ```bash
   pytest
   ```

## License

MIT License

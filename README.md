# Django Chain
A Django library for seamless LangChain integration, making it easy to add LLM capabilities to your Django applications.

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/Brian-Kariu/django-chain/graph/badge.svg?token=C2C53JBPKO)](https://codecov.io/gh/Brian-Kariu/django-chain)
[![ci](https://github.com/Brian-Kariu/django-chain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Brian-Kariu/cookiecutter-airflow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

- Easy integration with existing Django models and views
- Built-in utilities for common LLM tasks
- Type-safe and well-documented API
- Comprehensive test coverage
- Production-ready with proper error handling

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

## Quick Start

```python
from django_chain.models import LLMChain
from django_chain.views import LLMView

# Create a chain
chain = LLMChain.objects.create(
    name="my_chain",
    prompt_template="Answer the following question: {question}",
    model_name="gpt-3.5-turbo"
)

# Use in a view
class MyLLMView(LLMView):
    chain = chain

    def get_prompt_context(self, request):
        return {"question": request.GET.get("question", "")}
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

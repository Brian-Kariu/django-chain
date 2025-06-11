# Django Chain

A Django library for seamless LangChain integration, making it easy to add LLM capabilities to your Django applications.

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

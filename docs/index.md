# Django Chain Documentation

Welcome to the official documentation for **Django Chain**!

Django Chain is a Django library that makes it easy to integrate Large Language Models (LLMs) into your Django applications using familiar Django patterns. It provides a comprehensive framework for building AI-powered applications with production-ready features.

## 🚀 **Key Features**

- **Django-native abstractions** for LLMs, prompts, and workflows
- **Database-backed** prompt and memory management with versioning
- **Multi-provider support** (OpenAI, Google, HuggingFace, and more)
- **Comprehensive logging** and interaction tracking
- **Vector store integration** for RAG (Retrieval-Augmented Generation)
- **Chat session management** with persistent history
- **Async support** and Celery integration for background tasks
- **Production-ready** error handling and monitoring
- **Extensible architecture** for custom workflows and providers
- **Type-safe API** with comprehensive test coverage

## 🎯 **Who Should Use Django Chain?**

- **Django developers** who want to add LLM-powered features (chatbots, summarization, content generation) to their applications
- **AI/ML teams** seeking a maintainable, scalable, and testable LLM integration solution
- **Product teams** building conversational interfaces or AI-enhanced user experiences
- **Enterprise developers** who need production-ready LLM integration with proper logging and monitoring

## 🏗️ **Core Concepts**

- **Prompts**: Configurable templates for generating LLM prompts (PromptTemplate, ChatPromptTemplate, etc.)
- **Workflows**: Sequences of ordered steps that orchestrate LLM interactions (prompt → LLM → parser → output)
- **Chat Sessions**: Persistent conversation management with user context and message history
- **Interaction Logs**: Comprehensive tracking of all LLM interactions with metadata and performance metrics
- **Vector Stores**: Document storage and semantic search capabilities for RAG applications

## 📋 **Quick Example**

```python
from django_chain.models import Prompt, Workflow

# Create a prompt template
prompt = Prompt.objects.create(
    name="customer_support",
    prompt_template={
        "langchain_type": "PromptTemplate",
        "template": "You are a helpful customer support agent. Question: {question}",
        "input_variables": ["question"]
    },
    input_variables=["question"]
)

# Create a workflow
workflow = Workflow.objects.create(
    name="support_workflow",
    prompt=prompt,
    workflow_definition=[
        {"type": "prompt", "name": "customer_support"},
        {"type": "llm", "provider": "openai"},
        {"type": "parser", "parser_type": "str"}
    ]
)

# Execute the workflow
chain = workflow.to_langchain_chain()
result = chain.invoke({"question": "How do I reset my password?"})
```

---

## 📚 **Documentation Sections**

### **Getting Started**
- [Installation Guide](introduction/installation.md) - Set up Django Chain in your project
- [Design & Architecture](introduction/design.md) - Understand the core architecture
- [Configuration](introduction/configuration.md) - Configure settings and providers

### **Usage & Examples**
- [Tutorials](usage/tutorials.md) - Step-by-step guides for common use cases
- [How-to Guides](usage/how-to-guides.md) - Practical solutions for specific tasks
- [Examples](usage/examples.md) - Code examples and patterns
- [Testing](usage/testing.md) - Testing your LLM integrations

### **API Reference**
- [Models](api/models.md) - Core Django models (Prompt, Workflow, ChatSession, etc.)
- [Views](api/views.md) - REST API endpoints and views
- [Utilities](api/utilities.md) - Helper functions and utilities
- [Providers](api/providers.md) - LLM provider integrations
- [Mixins](api/mixins.md) - Reusable view mixins

### **Advanced Topics**
- [Custom Providers](advanced/custom-providers.md) - Build custom LLM providers
- [Vector Stores](advanced/vector-stores.md) - RAG and semantic search
- [Performance](advanced/performance.md) - Optimization and scaling
- [Security](advanced/security.md) - Best practices and considerations

### **Development**
- [Contributing](contributing/README.md) - How to contribute to the project
- [Testing Strategy](contributing/testing.md) - Running and writing tests
- [Release Process](contributing/releases.md) - Release and versioning

---

## 🔗 **Quick Links**

- **GitHub Repository**: [https://github.com/Brian-Kariu/django-chain](https://github.com/Brian-Kariu/django-chain)
- **PyPI Package**: [https://pypi.org/project/django-chain/](https://pypi.org/project/django-chain/)
- **Issue Tracker**: [https://github.com/Brian-Kariu/django-chain/issues](https://github.com/Brian-Kariu/django-chain/issues)
- **Vanilla Django Example**: [Complete working example project](https://github.com/Brian-Kariu/django-chain/tree/main/examples/vanilla_django)

## ⚠️ **Project Status**

> **Alpha Release**: Django Chain is currently in early alpha (v0.2.0a3). While the core functionality is stable and tested, you may encounter breaking changes between versions. We recommend pinning to specific versions in production and thoroughly testing upgrades.

Use the navigation on the left to explore each section in detail.

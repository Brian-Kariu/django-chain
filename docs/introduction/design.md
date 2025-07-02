# Design & Architecture

Django Chain is designed to provide seamless, Django-native integration with Large Language Models (LLMs) using the LangChain framework. Its architecture emphasizes modularity, extensibility, and production readiness.

## Core Principles
- **Django-Native Abstraction:** Use familiar Django models, views, and patterns.
- **Progressive Enhancement:** Easily add LLM features to existing Django projects.
- **Performance & Scalability:** Async support and Celery for background tasks.
- **Observability:** Built-in logging, error handling, and monitoring hooks.
- **Modularity:** Extensible components for LLMs, prompts, memory, and vector stores.

## Architecture Overview
```
Django App (django_chain)
├── Models: Prompt, Workflow, ChatSession, ChatMessage, InteractionLog, etc.
├── Services: LLMClient, PromptManager, ChainExecutor, VectorStoreManager
├── Views: REST-style endpoints for prompt/workflow management and LLM execution
├── Memory: Utilities for conversation history and memory management
├── Providers: Integrations for OpenAI, Google, Fake (testing), etc.
├── Vector DB Integrations: pgvector and future RAG support
├── Signals & Exceptions: Custom error handling and event hooks
```

## Key Components
- **Prompt & Workflow Models:** Store and version prompts and workflows in the database.
- **LLMClient & Services:** Abstract LLM provider logic and chain execution.
- **Memory Management:** Persist chat history and hydrate LangChain memory objects.
- **API Views:** Expose endpoints for prompt/workflow CRUD, LLM calls, and vector search.
- **Extensibility:** Add new providers, memory types, or vector stores as needed.

See the [API Reference](../api/intro.md) for details on each module.

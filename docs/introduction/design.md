# Design & Architecture

Django Chain is designed to provide seamless, Django-native integration with Large Language Models (LLMs) using the LangChain framework. Its architecture emphasizes modularity, extensibility, production readiness, and developer experience.

## 🎯 **Core Principles**

- **Django-Native Abstraction**: Use familiar Django models, views, and patterns for LLM integration
- **Progressive Enhancement**: Easily add LLM features to existing Django projects without major refactoring
- **Production Ready**: Built-in logging, error handling, monitoring, and security features
- **Performance & Scalability**: Async support, caching, and background task integration
- **Developer Experience**: Type-safe APIs, comprehensive testing, and clear documentation
- **Modularity**: Extensible components for LLMs, prompts, memory, vector stores, and workflows

## 🏗️ **Architecture Overview**

```
Django Chain Architecture
┌─────────────────────────────────────────────────────────────┐
│                    Django Application                       │
├─────────────────────────────────────────────────────────────┤
│ Models Layer                                                │
│ ├── Prompt (templates & versioning)                         │
│ ├── Workflow (orchestration definitions)                    │
│ ├── ChatSession (conversation management)                   │
│ ├── ChatHistory (message storage)                           │
│ └── InteractionLog (comprehensive logging)                  │
├─────────────────────────────────────────────────────────────┤
│ Views & API Layer                                           │
│ ├── REST endpoints (CRUD operations)                        │
│ ├── Chat management views                                   │
│ ├── Workflow execution views                                │
│ └── Vector search endpoints                                 │
├─────────────────────────────────────────────────────────────┤
│ Services & Business Logic                                   │
│ ├── VectorStoreManager (document & search)                  │
│ └── Configuration management (app_settings)                 │
├─────────────────────────────────────────────────────────────┤
│ Utilities Layer                                             │
│ ├── LLM Client utilities (provider abstraction)             │
│ ├── Memory Manager (conversation persistence)               │
│ ├── Workflow Processor (LangChain integration)              │
│ └── Logging Handler (comprehensive tracking)                │
├─────────────────────────────────────────────────────────────┤
│ Providers Layer                                             │
│ ├── OpenAI integration                                      │
│ ├── Google AI integration                                   │
│ ├── HuggingFace integration                                 │
│ ├── Vector Store providers (pgvector, chroma, pinecone)     │
│ └── Fake provider (testing)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 **Core Components**

### **Models Layer**

**Purpose**: Django models that provide database-backed storage for LLM components

- **`Prompt`**: Configurable templates for LLM prompts with versioning support
  - Supports PromptTemplate, ChatPromptTemplate, and custom templates
  - Input variable validation and template rendering
  - Version control and activation/deactivation

- **`Workflow`**: Sequences of steps that orchestrate LLM interactions
  - JSON-defined workflow steps (prompt → LLM → parser)
  - Integration with LangChain's Runnable interface
  - Support for complex multi-step AI workflows

- **`ChatSession`**: Persistent conversation management
  - User-associated chat sessions with unique identifiers
  - Session archiving and activation controls
  - Integration with workflow execution

- **`ChatHistory`**: Message storage with role-based organization
  - Support for user, assistant, and system messages
  - Soft deletion for privacy compliance
  - Chronological message ordering

- **`InteractionLog`**: Comprehensive tracking of all LLM interactions
  - Token usage tracking for cost analysis
  - Performance metrics (latency, success rates)
  - Error tracking and debugging information
  - Sensitive data sanitization

### **Views & API Layer**

**Purpose**: REST API endpoints and Django views for managing LLM components

- **CRUD Operations**: Full create, read, update, delete operations for all models
- **Workflow Execution**: Endpoints for executing workflows with logging
- **Chat Management**: Session creation, message handling, and history retrieval
- **Vector Operations**: Document storage, indexing, and semantic search
- **Monitoring**: Interaction logs and analytics endpoints

### **Services Layer**

**Purpose**: Business logic and service classes for complex operations

- **`VectorStoreManager`**: Manages document storage and semantic search
  - Multi-provider support (pgvector, chroma, pinecone)
  - Document indexing and metadata management
  - Similarity search with configurable parameters

- **Configuration Management**: Django App Settings pattern
  - Centralized configuration with validation
  - Environment-specific settings support
  - Runtime configuration updates

### **Utilities Layer**

**Purpose**: Helper functions and utilities for LLM operations

- **LLM Client Utilities**: Provider-agnostic LLM interaction
  - Multi-provider support with consistent interfaces
  - Comprehensive logging with sensitive data sanitization
  - Error handling and retry logic

- **Memory Manager**: Conversation memory persistence
  - Multiple memory types (buffer, buffer_window, summary)
  - Storage backends (in-memory, PostgreSQL, Redis)
  - Integration with LangChain memory objects

- **Workflow Processor**: Converts workflows to LangChain chains
  - Dynamic workflow definition processing
  - LangChain Runnable component integration
  - Step-by-step execution with logging

### **Providers Layer**

**Purpose**: Integration with various LLM and vector store providers

- **LLM Providers**: OpenAI, Google AI, HuggingFace, Fake (testing)
- **Vector Stores**: pgvector, chroma, pinecone
- **Extensible**: Easy to add new providers following established patterns

## 🔄 **Data Flow Architecture**

### **Workflow Execution Flow**

```
1. User Request → Django View
2. View → Workflow Model (database lookup)
3. Workflow → WorkflowProcessor (convert to LangChain)
4. WorkflowProcessor → LLM Provider (via utilities)
5. Provider Response → LoggingHandler (comprehensive logging)
6. LoggingHandler → InteractionLog (database storage)
7. Response → User (via Django view)
```

### **Chat Session Flow**

```
1. User Message → ChatDemoView
2. View → ChatSession (get or create)
3. View → ChatHistory (store user message)
4. View → Workflow Execution (process message)
5. LLM Response → ChatHistory (store assistant message)
6. Response → User Interface
```

### **Vector Store Flow**

```
1. Document Upload → VectorStoreDemo
2. Demo → VectorStoreManager (process document)
3. Manager → Embedding Provider (generate vectors)
4. Manager → Vector Store (pgvector/chroma/pinecone)
5. Search Query → Similarity Search
6. Results → User Interface
```

## 🔧 **Integration Patterns**

### **LangChain Integration**

Django Chain seamlessly integrates with LangChain components:

- **Runnable Interface**: Workflows implement LangChain's Runnable pattern
- **Memory Objects**: Django models hydrate LangChain memory objects
- **Prompt Templates**: Database-stored prompts become LangChain templates
- **Callback Handlers**: Custom logging handlers capture all interactions

### **Django Integration**

Built using Django best practices:

- **Model-View-Template**: Clear separation of concerns
- **Django ORM**: Database operations through Django models
- **Django Settings**: Configuration through Django's settings system
- **Django Signals**: Event-driven architecture for extensibility
- **Django Admin**: Built-in administrative interface

### **Provider Integration**

Consistent patterns for adding new providers:

```python
# Example provider integration
from django_chain.providers import register_llm_provider

@register_llm_provider("custom_provider")
def create_custom_llm(config):
    return CustomLLM(**config)
```

## 🛡️ **Security Architecture**

### **Sensitive Data Handling**

- **API Key Protection**: Never log or expose API keys
- **Data Sanitization**: Automatic removal of sensitive patterns
- **Environment Variables**: Secure configuration management
- **Access Controls**: User-based data isolation

### **Privacy Compliance**

- **Soft Deletion**: ChatHistory supports soft deletion
- **Data Retention**: Configurable retention policies
- **User Isolation**: Multi-tenant data separation
- **Audit Logging**: Comprehensive interaction tracking

## 📊 **Monitoring & Observability**

### **Comprehensive Logging**

- **Interaction Tracking**: Every LLM call logged with metadata
- **Performance Metrics**: Latency, token usage, success rates
- **Error Tracking**: Detailed error categorization and debugging
- **Cost Analysis**: Token usage tracking for billing insights

### **Health Monitoring**

- **Provider Health**: Monitor LLM provider availability
- **Database Health**: Track model operations and performance
- **Cache Health**: Monitor caching effectiveness
- **Vector Store Health**: Track indexing and search performance

## 🚀 **Performance Architecture**

### **Caching Strategy**

- **LLM Response Caching**: Reduce API calls for repeated queries
- **Model Caching**: Cache frequently accessed prompts and workflows
- **Vector Caching**: Cache embedding computations
- **Configuration Caching**: Cache app settings for performance

### **Async Support**

- **Async Views**: Non-blocking LLM operations
- **Background Tasks**: Celery integration for long-running operations
- **Streaming Responses**: Support for streaming LLM responses
- **Connection Pooling**: Efficient provider connection management

### **Scalability Patterns**

- **Horizontal Scaling**: Stateless design for multi-instance deployment
- **Database Optimization**: Efficient queries with proper indexing
- **Provider Load Balancing**: Distribute load across multiple providers
- **Caching Layers**: Multiple caching strategies for different use cases

## 🔌 **Extensibility Architecture**

### **Plugin System**

- **Custom Providers**: Easy integration of new LLM providers
- **Custom Memory Types**: Extensible memory management
- **Custom Parsers**: Custom output parsing logic
- **Custom Workflows**: Complex workflow orchestration

### **Event System**

- **Django Signals**: Hook into model lifecycle events
- **Callback Handlers**: Custom logging and monitoring
- **Middleware**: Request/response processing hooks
- **Custom Validators**: Extensible validation logic

## 📋 **Design Decisions**

### **Why Django App Settings Pattern?**

- **Familiar**: Similar to Django REST Framework's approach
- **Testable**: Easy to override settings in tests
- **Validatable**: Comprehensive validation on startup
- **Extensible**: Easy to add new configuration options

### **Why Database-Backed Models?**

- **Persistence**: Prompts and workflows survive application restarts
- **Versioning**: Track changes and enable rollbacks
- **Collaboration**: Multiple developers can share configurations
- **Administration**: Django admin interface for management

### **Why Provider Abstraction?**

- **Flexibility**: Switch between providers without code changes
- **Testing**: Use fake providers for testing without API costs
- **Reliability**: Fallback to alternative providers
- **Cost Optimization**: Use different providers for different use cases

## 📚 **Next Steps**

To learn more about Django Chain's architecture:

- **[Configuration Guide](configuration.md)**: Detailed configuration options
- **[API Reference](../api/intro.md)**: Complete API documentation
- **[Tutorials](../usage/tutorials.md)**: Practical implementation guides
- **[Example Project](https://github.com/Brian-Kariu/django-chain/tree/main/examples/vanilla_django)**: Complete working implementation

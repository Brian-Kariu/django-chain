# Django-Chain Vanilla Django Example Project

This comprehensive example project demonstrates all major features of **django-chain**, a Django library for seamless Large Language Model (LLM) integration. It serves as both a testing environment for key features and a practical guide for developers looking to integrate LLM capabilities into their Django applications.

## 🌟 Features Demonstrated

### 📝 **Prompt Management**
- Create and manage prompt templates
- Support for both simple prompts and complex chat prompt templates
- Template versioning and activation/deactivation
- Variable validation and LangChain integration

### 🔄 **Workflow Orchestration**
- Build complex AI workflows by chaining prompts, LLMs, and parsers
- Execute workflows with comprehensive logging and error handling
- Support for different workflow types (simple, with parsers, custom)
- Integration with chat sessions and user context

### 💬 **Chat Session Management**
- Persistent chat sessions with unique session IDs
- Message history storage with role-based organization
- User association and authentication integration
- Chat analytics and session archiving

### 🗄️ **Vector Store Operations**
- Document storage and indexing
- Semantic search capabilities
- Integration with pgvector for PostgreSQL
- Document management with metadata

### 🔧 **LLM Provider Testing**
- Test different LLM providers (OpenAI, Google, HuggingFace, Fake)
- Chat model and embedding model testing
- Provider configuration validation
- Response format analysis

### 📊 **Interaction Logging & Analytics**
- Comprehensive logging of all LLM interactions
- Token usage tracking for cost analysis
- Performance metrics (latency, success rates)
- Error tracking and debugging information

### 🛠️ **Custom Workflow Building**
- Programmatic workflow creation using utility functions
- Custom model integration patterns
- Advanced workflow composition examples
- Error handling and recovery patterns

## 🚀 Quick Start

### 1. **Navigate to the Project**
```bash
cd examples/vanilla_django
```

### 2. **Install Dependencies**
```bash
pip install -r ../../requirements.txt
```

### 3. **Run Migrations**
```bash
python manage.py migrate
```

### 4. **Start the Development Server**
```bash
python manage.py runserver
```

### 5. **Access the Application**
- **Main Dashboard**: http://127.0.0.1:8000/example/
- **API Overview**: http://127.0.0.1:8000/example/api/
- **Django Admin**: http://127.0.0.1:8000/admin/

## 📱 User Interface

The project includes a modern, responsive web interface built with Bootstrap 5:

- **Dashboard**: Overview of all features with quick actions
- **Interactive Demos**: User-friendly interfaces for each feature
- **Real-time Testing**: Live LLM interaction testing
- **Visual Analytics**: Charts and metrics for interaction logs

## 🔗 API Endpoints

### **Core Django-Chain APIs**
```
GET  /django-chain/prompts/              # List prompts
POST /django-chain/prompts/              # Create prompt
GET  /django-chain/workflows/            # List workflows
POST /django-chain/workflows/            # Create workflow
POST /django-chain/workflows/{name}/execute/  # Execute workflow
GET  /django-chain/chat/sessions/        # List chat sessions
POST /django-chain/chat/sessions/        # Create chat session
GET  /django-chain/interactions/         # List interaction logs
```

### **Example Application APIs**
```
GET  /example/                           # Dashboard
GET  /example/api/                       # API documentation
GET  /example/prompts/                   # Prompt examples interface
POST /example/prompts/                   # Create example prompts
GET  /example/workflows/                 # Workflow examples interface
POST /example/workflows/                 # Create example workflows
POST /example/workflows/execute/         # Execute workflows
GET  /example/chat/                      # Chat demo interface
POST /example/chat/                      # Send chat messages
GET  /example/vector/                    # Vector store interface
POST /example/vector/                    # Vector operations
GET  /example/llm-test/                  # LLM testing interface
POST /example/llm-test/                  # Test LLM providers
GET  /example/logs/                      # Interaction logs interface
POST /example/errors/                    # Error handling demos
GET  /example/custom-workflow/           # Custom workflow builder
```

## 📋 Example Usage Patterns

### **1. Create a Simple Prompt**
```python
from django_chain.models import Prompt

prompt = Prompt.objects.create(
    name="customer_service_prompt",
    prompt_template={
        "langchain_type": "PromptTemplate",
        "template": "You are a helpful customer service assistant. User question: {question}",
        "input_variables": ["question"]
    },
    input_variables=["question"],
    is_active=True
)
```

### **2. Build a Workflow**
```python
from django_chain.models import Workflow

workflow = Workflow.objects.create(
    name="customer_service_workflow",
    description="Handle customer service inquiries",
    prompt=prompt,
    workflow_definition=[
        {"type": "prompt", "name": "customer_service_prompt"},
        {"type": "llm", "provider": "openai"},
        {"type": "parser", "parser_type": "str"}
    ],
    is_active=True
)
```

### **3. Execute a Workflow**
```python
# Execute with logging
chain = workflow.to_langchain_chain(
    log="true",
    llm_config={
        "DEFAULT_LLM_PROVIDER": "openai",
        "DEFAULT_CHAT_MODEL": {"name": "gpt-3.5-turbo"},
    }
)

result = chain.invoke({"question": "How can I return a product?"})
```

### **4. Manage Chat Sessions**
```python
from django_chain.models import ChatSession, ChatHistory

# Create session
session = ChatSession.objects.create(
    session_id="unique-session-id",
    title="Customer Support Chat",
    user=request.user
)

# Add messages
ChatHistory.objects.create(
    session=session,
    content="Hello, I need help with my order",
    role="USER"
)
```

### **5. Use Vector Store**
```python
from django_chain.services.vector_store_manager import VectorStoreManager

manager = VectorStoreManager()

# Add documents
manager.add_documents_to_store([
    "Django-Chain is a powerful LLM integration library",
    "It provides seamless integration with Django applications"
])

# Search
results = manager.search_documents("LLM integration", k=5)
```

## 🏗️ Architecture Patterns

### **Custom Model Integration**
The example app demonstrates how to create custom models that integrate with django-chain:

- `ExampleDocument`: Document storage with vector indexing
- `ExampleWorkflowExecution`: Custom execution tracking
- `ExampleChatAnalytics`: Chat session analytics
- `ExamplePromptTemplate`: Domain-specific prompt templates

### **Service Layer Patterns**
Examples of how to build services on top of django-chain:

```python
class CustomWorkflowService:
    def __init__(self):
        self.workflow_processor = WorkflowProcessor()

    def build_custom_workflow(self, steps, config):
        # Custom workflow building logic
        pass

    def execute_with_analytics(self, workflow, input_data):
        # Execute and track analytics
        pass
```

### **Error Handling Patterns**
Comprehensive error handling examples:

```python
try:
    result = workflow.execute(input_data)
except PromptValidationError as e:
    # Handle prompt validation errors
    logger.error(f"Prompt validation failed: {e}")
except WorkflowValidationError as e:
    # Handle workflow validation errors
    logger.error(f"Workflow validation failed: {e}")
except LLMProviderAPIError as e:
    # Handle LLM provider errors
    logger.error(f"LLM provider error: {e}")
```

## 🧪 Testing Features

### **Interactive Testing**
- **Dashboard Quick Actions**: Create prompts and workflows with one click
- **Chat Demo**: Real-time conversation testing
- **LLM Provider Testing**: Test different providers and models
- **Vector Search Testing**: Add documents and test semantic search

### **API Testing**
Use the provided API endpoints to test programmatically:

```bash
# Create a prompt
curl -X POST http://127.0.0.1:8000/example/prompts/ \
  -H "Content-Type: application/json" \
  -d '{"example_type": "simple"}'

# Execute a workflow
curl -X POST http://127.0.0.1:8000/example/workflows/execute/ \
  -H "Content-Type: application/json" \
  -d '{"workflow_name": "example_workflow", "input": {"question": "Hello"}}'
```

## 🔧 Configuration

The project uses django-chain's configuration system:

```python
# settings.py
DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "fake",  # Change to "openai" for real testing
    "DEFAULT_CHAT_MODEL": {
        "name": "fake-model",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": "fake key",  # Set real API key for production
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
}
```

## 📚 Learning Resources

### **Code Examples**
- `views.py`: Comprehensive view examples for all features
- `models.py`: Custom model integration patterns
- `templates/`: Modern UI implementation examples

### **Key Files to Study**
- `views.py`: Complete feature implementation
- `models.py`: Custom model patterns
- `urls.py`: URL configuration patterns
- `templates/example/`: Frontend integration examples

## 🤝 Contributing

This example project is designed to be:
- **Educational**: Clear, well-documented code
- **Comprehensive**: Covers all major features
- **Practical**: Real-world usage patterns
- **Extensible**: Easy to add new examples

Feel free to extend the examples or add new demonstration features!

## 📄 License

This example project follows the same license as the main django-chain library.

---

**Happy coding with Django-Chain! 🚀**

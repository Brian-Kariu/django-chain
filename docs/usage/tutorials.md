# Tutorials

Learn Django Chain with comprehensive step-by-step tutorials covering all major features.

## 🚀 **Quick Start Tutorial**

### **Goal**: Create a simple customer support chatbot in 15 minutes

#### **Step 1: Setup Django Chain**

First, install and configure Django Chain in your project:

```bash
pip install "django-chain[openai]"
```

Add to your `settings.py`:

```python
# settings.py
INSTALLED_APPS = [
    # ... your other apps
    'django_chain',
]

DJANGO_LLM_SETTINGS = {
    "DEFAULT_LLM_PROVIDER": "openai",
    "DEFAULT_CHAT_MODEL": {
        "name": "gpt-3.5-turbo",
        "temperature": 0.3,  # Less random for support
        "max_tokens": 512,
        "api_key": "your-openai-api-key",
    },
    "ENABLE_LLM_LOGGING": True,
}
```

Run migrations:

```bash
python manage.py migrate django_chain
```

#### **Step 2: Create a Prompt Template**

```python
# In Django shell: python manage.py shell
from django_chain.models import Prompt

support_prompt = Prompt.objects.create(
    name="customer_support",
    prompt_template={
        "langchain_type": "PromptTemplate",
        "template": """You are a helpful customer support agent for TechCorp.

Guidelines:
- Be polite and professional
- Provide clear, concise answers
- If you don't know something, offer to escalate to human support
- Always end with asking if there's anything else you can help with

Customer question: {question}""",
        "input_variables": ["question"]
    },
    input_variables=["question"],
    description="Customer support prompt template",
    is_active=True
)
```

#### **Step 3: Create a Workflow**

```python
from django_chain.models import Workflow

support_workflow = Workflow.objects.create(
    name="customer_support_workflow",
    description="Handle customer support inquiries",
    prompt=support_prompt,
    workflow_definition=[
        {
            "type": "prompt",
            "name": "customer_support"
        },
        {
            "type": "llm",
            "provider": "openai"
        },
        {
            "type": "parser",
            "parser_type": "str"
        }
    ],
    is_active=True
)
```

#### **Step 4: Test the Workflow**

```python
# Execute the workflow
chain = support_workflow.to_langchain_chain()
result = chain.invoke({
    "question": "How do I reset my password?"
})

print(result)
# Output: "To reset your password, please follow these steps..."
```

#### **Step 5: Create a Simple View**

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

from django_chain.models import Workflow

@csrf_exempt
@require_http_methods(["POST"])
def customer_support_chat(request):
    try:
        data = json.loads(request.body)
        question = data.get('question', '')

        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)

        # Get the workflow
        workflow = Workflow.objects.get(name="customer_support_workflow")

        # Execute with logging
        chain = workflow.to_langchain_chain(log="true")
        result = chain.invoke({"question": question})

        return JsonResponse({
            'answer': result,
            'success': True
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500)
```

**Congratulations!** You now have a working customer support chatbot with Django Chain.

---

## 💬 **Chat Session Tutorial**

### **Goal**: Build a persistent chat interface with conversation memory

#### **Step 1: Create Chat Models**

Django Chain includes built-in chat session management:

```python
from django_chain.models import ChatSession, ChatHistory
from django.contrib.auth.models import User

# Create a chat session
user = User.objects.get(username='your_username')  # or request.user
session = ChatSession.objects.create(
    title="Customer Support Chat",
    user=user,
    # session_id is automatically generated as UUID
)
```

#### **Step 2: Enhanced Chat View**

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
import uuid

from django_chain.models import Workflow, ChatSession, ChatHistory

@csrf_exempt
@login_required
def persistent_chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')
        session_id = data.get('session_id')

        # Get or create chat session
        if session_id:
            try:
                session = ChatSession.objects.get(
                    session_id=session_id,
                    user=request.user
                )
            except ChatSession.DoesNotExist:
                return JsonResponse({'error': 'Session not found'}, status=404)
        else:
            session = ChatSession.objects.create(
                title=f"Chat {message[:30]}...",
                user=request.user
                # session_id is automatically generated as UUID
            )

        # Store user message
        user_message = ChatHistory.objects.create(
            session=session,
            role="user",
            content=message,
            metadata={"timestamp": timezone.now().isoformat()}
        )

        # Get workflow and execute
        workflow = Workflow.objects.get(name="customer_support_workflow")
        chain = workflow.to_langchain_chain(log="true")

        # Include conversation context
        recent_messages = ChatHistory.objects.filter(
            session=session
        ).order_by('-created_at')[:10]

        context = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in reversed(recent_messages)
        ])

        result = chain.invoke({
            "question": message,
            "context": context
        })

        # Store assistant response
        assistant_message = ChatHistory.objects.create(
            session=session,
            role="assistant",
            content=result,
            metadata={"model": "gpt-3.5-turbo"}
        )

        return JsonResponse({
            'response': result,
            'session_id': session.session_id,
            'message_id': assistant_message.id
        })

    elif request.method == 'GET':
        # Get chat history
        session_id = request.GET.get('session_id')
        if not session_id:
            return JsonResponse({'error': 'Session ID required'}, status=400)

        try:
            session = ChatSession.objects.get(
                session_id=session_id,
                user=request.user
            )
            messages = ChatHistory.objects.filter(
                session=session
            ).order_by('created_at')

            return JsonResponse({
                'session_id': session.session_id,
                'title': session.title,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'created_at': msg.created_at.isoformat()
                    }
                    for msg in messages
                ]
            })
        except ChatSession.DoesNotExist:
            return JsonResponse({'error': 'Session not found'}, status=404)
```

#### **Step 3: Frontend Integration**

```javascript
// chat.js
class ChatInterface {
    constructor() {
        this.sessionId = localStorage.getItem('chat_session_id');
        this.loadChatHistory();
    }

    async sendMessage(message) {
        const response = await fetch('/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                message: message,
                session_id: this.sessionId
            })
        });

        const data = await response.json();

        if (data.session_id) {
            this.sessionId = data.session_id;
            localStorage.setItem('chat_session_id', this.sessionId);
        }

        this.displayMessage('user', message);
        this.displayMessage('assistant', data.response);

        return data;
    }

    async loadChatHistory() {
        if (!this.sessionId) return;

        const response = await fetch(`/chat/?session_id=${this.sessionId}`);
        const data = await response.json();

        data.messages.forEach(msg => {
            this.displayMessage(msg.role, msg.content);
        });
    }

    displayMessage(role, content) {
        const chatContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.textContent = content;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Initialize chat
const chat = new ChatInterface();
```

---

## 🔍 **Vector Store Tutorial**

### **Goal**: Build a knowledge base with semantic search

#### **Step 1: Setup Vector Store**

Configure pgvector in your settings:

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db',
        'USER': 'your_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

DJANGO_LLM_SETTINGS = {
    # ... other settings
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "openai",
        "name": "text-embedding-ada-002",
        "api_key": "your-openai-api-key",
    },
    "VECTOR_STORE": {
        "TYPE": "pgvector",
        "PGVECTOR_COLLECTION_NAME": "knowledge_base",
    },
}
```

Create the pgvector extension in PostgreSQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

#### **Step 2: Create Document Model**

```python
# models.py
from django.db import models

class KnowledgeDocument(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    category = models.CharField(max_length=100)
    tags = models.JSONField(default=list)
    is_indexed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

#### **Step 3: Document Indexing**

```python
from django_chain.services.vector_store_manager import VectorStoreManager
from django_chain.utils.llm_client import create_llm_embedding_client

def index_document(document):
    """Index a document in the vector store."""

    # Create embedding client
    embedding_client = create_llm_embedding_client()

    # Initialize vector store manager
    vector_manager = VectorStoreManager()

    # Prepare document data
    texts = [document.content]
    metadatas = [{
        "title": document.title,
        "category": document.category,
        "tags": document.tags,
        "doc_id": document.id
    }]

    # Add to vector store
    vector_manager.add_documents_to_store(
        texts=texts,
        metadatas=metadatas,
        embedding_function=embedding_client
    )

    # Mark as indexed
    document.is_indexed = True
    document.save()

# Index existing documents
for doc in KnowledgeDocument.objects.filter(is_indexed=False):
    index_document(doc)
```

#### **Step 4: Semantic Search**

```python
def search_knowledge_base(query, limit=5):
    """Search the knowledge base using semantic similarity."""

    embedding_client = create_llm_embedding_client()
    vector_manager = VectorStoreManager()

    # Perform similarity search
    results = vector_manager.search_documents(
        query=query,
        k=limit,
        embedding_function=embedding_client
    )

    return [
        {
            "content": result.page_content,
            "metadata": result.metadata,
            "score": getattr(result, 'score', 0.0)
        }
        for result in results
    ]

# Example usage
results = search_knowledge_base("How to reset password")
for result in results:
    print(f"Title: {result['metadata']['title']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Score: {result['score']}\n")
```

#### **Step 5: RAG-Enhanced Chat**

```python
def rag_enhanced_chat(question):
    """Chat with knowledge base context."""

    # Search knowledge base
    knowledge = search_knowledge_base(question, limit=3)

    # Create context from search results
    context = "\n\n".join([
        f"Document: {kb['metadata']['title']}\nContent: {kb['content']}"
        for kb in knowledge
    ])

    # Create RAG prompt
    rag_prompt = Prompt.objects.create(
        name="rag_support",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": """You are a helpful assistant with access to a knowledge base.

Use the following context to answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:""",
            "input_variables": ["context", "question"]
        },
        input_variables=["context", "question"]
    )

    # Create workflow
    workflow = Workflow.objects.create(
        name="rag_workflow",
        prompt=rag_prompt,
        workflow_definition=[
            {"type": "prompt", "name": "rag_support"},
            {"type": "llm", "provider": "openai"},
            {"type": "parser", "parser_type": "str"}
        ]
    )

    # Execute
    chain = workflow.to_langchain_chain()
    result = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": result,
        "sources": [kb['metadata']['title'] for kb in knowledge]
    }

# Example usage
response = rag_enhanced_chat("How do I reset my password?")
print(f"Answer: {response['answer']}")
print(f"Sources: {', '.join(response['sources'])}")
```

---

## 📊 **Analytics Tutorial**

### **Goal**: Monitor and analyze LLM interactions

#### **Step 1: Access Interaction Logs**

```python
from django_chain.models import InteractionLog
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta

def get_usage_analytics(days=7):
    """Get LLM usage analytics for the last N days."""

    since = timezone.now() - timedelta(days=days)

    logs = InteractionLog.objects.filter(
        created_at__gte=since
    )

    analytics = {
        "total_interactions": logs.count(),
        "successful_interactions": logs.filter(
            status="SUCCESS"
        ).count(),
        "failed_interactions": logs.filter(
            status="FAILURE"
        ).count(),
        "total_input_tokens": logs.aggregate(
            total=models.Sum('input_tokens')
        )['total'] or 0,
        "total_output_tokens": logs.aggregate(
            total=models.Sum('output_tokens')
        )['total'] or 0,
        "average_latency": logs.aggregate(
            avg=Avg('latency_ms')
        )['avg'] or 0,
        "workflows_used": logs.values('workflow__name').annotate(
            count=Count('id')
        ).order_by('-count')
    }

    return analytics

# Example usage
analytics = get_usage_analytics(30)  # Last 30 days
print(f"Total interactions: {analytics['total_interactions']}")
print(f"Success rate: {analytics['successful_interactions'] / analytics['total_interactions'] * 100:.1f}%")
print(f"Total tokens used: {analytics['total_input_tokens'] + analytics['total_output_tokens']}")
```

#### **Step 2: Cost Tracking**

```python
def calculate_costs(days=30):
    """Calculate estimated costs for OpenAI usage."""

    since = timezone.now() - timedelta(days=days)

    logs = InteractionLog.objects.filter(
        created_at__gte=since,
        model_name__startswith="gpt"  # OpenAI models
    )

    # OpenAI pricing (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0}
    }

    total_cost = 0
    cost_breakdown = {}

    for log in logs:
        model = log.model_name
        if model in pricing:
            input_cost = (log.input_tokens / 1000) * pricing[model]["input"]
            output_cost = (log.output_tokens / 1000) * pricing[model]["output"]

            model_cost = input_cost + output_cost
            total_cost += model_cost

            if model not in cost_breakdown:
                cost_breakdown[model] = {"cost": 0, "calls": 0}

            cost_breakdown[model]["cost"] += model_cost
            cost_breakdown[model]["calls"] += 1

    return {
        "total_cost": total_cost,
        "breakdown": cost_breakdown,
        "period_days": days
    }

# Example usage
costs = calculate_costs(30)
print(f"Total cost (30 days): ${costs['total_cost']:.2f}")
for model, data in costs['breakdown'].items():
    print(f"{model}: ${data['cost']:.2f} ({data['calls']} calls)")
```

#### **Step 3: Performance Dashboard**

```python
from django.shortcuts import render
from django.http import JsonResponse

def analytics_dashboard(request):
    """Analytics dashboard view."""

    if request.headers.get('accept') == 'application/json':
        # API endpoint for dashboard data
        days = int(request.GET.get('days', 7))

        analytics = get_usage_analytics(days)
        costs = calculate_costs(days)

        # Error analysis
        error_logs = InteractionLog.objects.filter(
            status="FAILURE",
            created_at__gte=timezone.now() - timedelta(days=days)
        ).values('error_message').annotate(
            count=Count('id')
        ).order_by('-count')[:10]

        return JsonResponse({
            "analytics": analytics,
            "costs": costs,
            "top_errors": list(error_logs),
            "period_days": days
        })

    return render(request, 'analytics_dashboard.html')
```

---

## 🧪 **Testing Tutorial**

### **Goal**: Test your LLM integrations effectively

#### **Step 1: Test Setup**

```python
# tests/test_workflows.py
from django.test import TestCase
from django.test.utils import override_settings
from unittest.mock import patch, MagicMock

from django_chain.models import Prompt, Workflow
from django_chain.config import app_settings

class WorkflowTestCase(TestCase):

    @override_settings(DJANGO_LLM_SETTINGS={
        'DEFAULT_LLM_PROVIDER': 'fake',
        'DEFAULT_CHAT_MODEL': {
            'name': 'fake-model',
            'api_key': 'fake-key',
        },
        'ENABLE_LLM_LOGGING': False,
    })
    def setUp(self):
        """Setup test data with fake provider."""
        app_settings.reload()  # Reload settings

        self.prompt = Prompt.objects.create(
            name="test_prompt",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": "Test question: {question}",
                "input_variables": ["question"]
            },
            input_variables=["question"]
        )

        self.workflow = Workflow.objects.create(
            name="test_workflow",
            prompt=self.prompt,
            workflow_definition=[
                {"type": "prompt", "name": "test_prompt"},
                {"type": "llm", "provider": "fake"},
                {"type": "parser", "parser_type": "str"}
            ]
        )
```

#### **Step 2: Mock LLM Responses**

```python
    @patch('django_chain.utils.llm_client.create_llm_chat_client')
    def test_workflow_execution(self, mock_create_client):
        """Test workflow execution with mocked LLM."""

        # Mock the LLM client
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Mocked response"
        mock_create_client.return_value = mock_llm

        # Execute workflow
        chain = self.workflow.to_langchain_chain()
        result = chain.invoke({"question": "Test question"})

        # Assertions
        self.assertEqual(result, "Mocked response")
        mock_create_client.assert_called_once()
        mock_llm.invoke.assert_called_once()
```

#### **Step 3: Integration Testing**

```python
    def test_real_llm_integration(self):
        """Integration test with real LLM (use sparingly)."""

        # Only run if REAL_LLM_TESTS environment variable is set
        import os
        if not os.getenv('REAL_LLM_TESTS'):
            self.skipTest("Real LLM tests disabled")

        # Use real OpenAI for integration testing
        with override_settings(DJANGO_LLM_SETTINGS={
            'DEFAULT_LLM_PROVIDER': 'openai',
            'DEFAULT_CHAT_MODEL': {
                'name': 'gpt-3.5-turbo',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'max_tokens': 50,  # Keep costs low
            },
        }):
            app_settings.reload()

            chain = self.workflow.to_langchain_chain()
            result = chain.invoke({"question": "What is 2+2?"})

            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
```

#### **Step 4: Performance Testing**

```python
import time
from django.test import TransactionTestCase

class PerformanceTestCase(TransactionTestCase):

    def test_workflow_performance(self):
        """Test workflow execution performance."""

        # Setup
        prompt = Prompt.objects.create(
            name="perf_test",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": "Question: {question}",
                "input_variables": ["question"]
            },
            input_variables=["question"]
        )

        workflow = Workflow.objects.create(
            name="perf_workflow",
            prompt=prompt,
            workflow_definition=[
                {"type": "prompt", "name": "perf_test"},
                {"type": "llm", "provider": "fake"},
                {"type": "parser", "parser_type": "str"}
            ]
        )

        # Performance test
        start_time = time.time()

        for i in range(100):
            chain = workflow.to_langchain_chain()
            result = chain.invoke({"question": f"Question {i}"})

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # Assert reasonable performance
        self.assertLess(avg_time, 0.1)  # Less than 100ms average
        print(f"Average execution time: {avg_time:.3f}s")
```

---

## 🎯 **Next Steps**

Now that you've completed these tutorials:

1. **Explore the [How-to Guides](how-to-guides.md)** for specific use cases
2. **Check out the [Vanilla Django Example](https://github.com/Brian-Kariu/django-chain/tree/main/examples/vanilla_django)** for a complete implementation
3. **Read the [API Reference](../api/intro.md)** for detailed documentation
4. **Learn about [Advanced Topics](../advanced/custom-providers.md)** for customization

## 💡 **Tips for Success**

- **Start with the fake provider** for development to avoid API costs
- **Use environment variables** for API keys and sensitive configuration
- **Enable logging** to understand what's happening in your workflows
- **Monitor token usage** to control costs
- **Test thoroughly** with mocked responses before using real LLMs
- **Keep prompts versioned** for easy rollbacks and A/B testing

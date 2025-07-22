# Code Examples

Practical code examples and patterns for common Django Chain use cases.

## 🏗️ **Basic Patterns**

### **Simple Prompt and Workflow**

```python
from django_chain.models import Prompt, Workflow

# Create a basic prompt
prompt = Prompt.objects.create(
    name="greeting",
    prompt_template={
        "langchain_type": "PromptTemplate",
        "template": "Say hello to {name} in a {style} way.",
        "input_variables": ["name", "style"]
    },
    input_variables=["name", "style"],
    description="Generate personalized greetings"
)

# Create a simple workflow
workflow = Workflow.objects.create(
    name="greeting_workflow",
    prompt=prompt,
    workflow_definition=[
        {"type": "prompt", "name": "greeting"},
        {"type": "llm", "provider": "openai"},
        {"type": "parser", "parser_type": "str"}
    ]
)

# Execute the workflow
chain = workflow.to_langchain_chain()
result = chain.invoke({
    "name": "Alice",
    "style": "professional"
})
print(result)  # "Good morning Alice, I hope you're having a productive day."
```

### **Chat Prompt Template**

```python
# More complex chat-based prompt
chat_prompt = Prompt.objects.create(
    name="customer_service_chat",
    prompt_template={
        "langchain_type": "ChatPromptTemplate",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful customer service representative for {company}. Be polite, professional, and helpful."
            },
            {
                "role": "human",
                "content": "{customer_message}"
            }
        ]
    },
    input_variables=["company", "customer_message"],
    description="Customer service chat template"
)

# Use in workflow
workflow = Workflow.objects.create(
    name="customer_service",
    prompt=chat_prompt,
    workflow_definition=[
        {"type": "prompt", "name": "customer_service_chat"},
        {"type": "llm", "provider": "openai"},
        {"type": "parser", "parser_type": "str"}
    ]
)

result = chain.invoke({
    "company": "TechCorp",
    "customer_message": "I'm having trouble logging into my account"
})
```

## 🔄 **Advanced Workflow Patterns**

### **Multi-Step Workflow with JSON Output**

```python
# Create a prompt that expects JSON output
analysis_prompt = Prompt.objects.create(
    name="sentiment_analysis",
    prompt_template={
        "langchain_type": "PromptTemplate",
        "template": """Analyze the sentiment of this text and return a JSON response.

Text: {text}

Return your analysis in this exact JSON format:
{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.95,
    "emotions": ["happy", "excited"],
    "summary": "Brief explanation"
}""",
        "input_variables": ["text"]
    },
    input_variables=["text"]
)

# Workflow with JSON parser
sentiment_workflow = Workflow.objects.create(
    name="sentiment_analysis",
    prompt=analysis_prompt,
    workflow_definition=[
        {"type": "prompt", "name": "sentiment_analysis"},
        {"type": "llm", "provider": "openai"},
        {"type": "parser", "parser_type": "json"}
    ]
)

# Execute and get structured data
chain = sentiment_workflow.to_langchain_chain()
result = chain.invoke({
    "text": "I absolutely love this new feature! It's amazing!"
})

print(result["sentiment"])    # "positive"
print(result["confidence"])   # 0.95
print(result["emotions"])     # ["happy", "excited"]
```

### **Conditional Workflow Logic**

```python
def create_smart_routing_workflow():
    """Create a workflow that routes queries to different specialists."""

    # Classification prompt
    classifier_prompt = Prompt.objects.create(
        name="query_classifier",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": """Classify this customer query into one of these categories:
- technical: Technical support issues
- billing: Billing and payment questions
- general: General information requests

Query: {query}

Category:""",
            "input_variables": ["query"]
        },
        input_variables=["query"]
    )

    # Technical support prompt
    tech_prompt = Prompt.objects.create(
        name="technical_support",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": """You are a technical support specialist. Help with this issue:

{query}

Provide step-by-step troubleshooting instructions.""",
            "input_variables": ["query"]
        },
        input_variables=["query"]
    )

    # Billing support prompt
    billing_prompt = Prompt.objects.create(
        name="billing_support",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": """You are a billing specialist. Help with this billing question:

{query}

Provide clear information about billing policies and next steps.""",
            "input_variables": ["query"]
        },
        input_variables=["query"]
    )

    # Create classification workflow
    classifier_workflow = Workflow.objects.create(
        name="query_classifier",
        prompt=classifier_prompt,
        workflow_definition=[
            {"type": "prompt", "name": "query_classifier"},
            {"type": "llm", "provider": "openai", "temperature": 0.1},
            {"type": "parser", "parser_type": "str"}
        ]
    )

    return classifier_workflow, tech_prompt, billing_prompt

def route_customer_query(query):
    """Route customer query to appropriate specialist."""

    classifier_workflow, tech_prompt, billing_prompt = create_smart_routing_workflow()

    # Classify the query
    classifier_chain = classifier_workflow.to_langchain_chain()
    category = classifier_chain.invoke({"query": query}).strip().lower()

    # Route to appropriate specialist
    if "technical" in category:
        specialist_workflow = Workflow.objects.create(
            name=f"tech_response_{uuid.uuid4()}",
            prompt=tech_prompt,
            workflow_definition=[
                {"type": "prompt", "name": "technical_support"},
                {"type": "llm", "provider": "openai"},
                {"type": "parser", "parser_type": "str"}
            ]
        )
    elif "billing" in category:
        specialist_workflow = Workflow.objects.create(
            name=f"billing_response_{uuid.uuid4()}",
            prompt=billing_prompt,
            workflow_definition=[
                {"type": "prompt", "name": "billing_support"},
                {"type": "llm", "provider": "openai"},
                {"type": "parser", "parser_type": "str"}
            ]
        )
    else:
        # Use general support
        specialist_workflow = Workflow.objects.get(name="customer_support_workflow")

    # Get specialist response
    specialist_chain = specialist_workflow.to_langchain_chain()
    response = specialist_chain.invoke({"query": query})

    return {
        "category": category,
        "response": response,
        "specialist": specialist_workflow.name
    }
```

## 💬 **Chat and Memory Patterns**

### **Conversation with Memory**

```python
from django_chain.utils.memory_manager import get_langchain_memory, save_messages_to_session
from django_chain.models import ChatSession, ChatHistory

def create_conversational_workflow():
    """Create a workflow that maintains conversation context."""

    conversation_prompt = Prompt.objects.create(
        name="conversational_assistant",
        prompt_template={
            "langchain_type": "ChatPromptTemplate",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the conversation history to provide contextual responses."
                },
                {
                    "role": "human",
                    "content": "{input}"
                }
            ]
        },
        input_variables=["input"]
    )

    return Workflow.objects.create(
        name="conversational_assistant",
        prompt=conversation_prompt,
        workflow_definition=[
            {"type": "prompt", "name": "conversational_assistant"},
            {"type": "llm", "provider": "openai"},
            {"type": "parser", "parser_type": "str"}
        ]
    )

def chat_with_memory(user, message, session_id=None):
    """Chat with persistent memory using Django models."""

    # Get or create session
    if session_id:
        try:
            session = ChatSession.objects.get(session_id=session_id, user=user)
        except ChatSession.DoesNotExist:
            session = ChatSession.objects.create(
                session_id=session_id,
                user=user,
                title=f"Chat Session {str(session_id)[:8]}"
            )
    else:
        session = ChatSession.objects.create(
            user=user,
            title="New Chat Session"
        )

    # Add user message to session
    session.add_message(message, role="USER")

    # Get workflow
    workflow = Workflow.objects.get(name="conversational_assistant", is_active=True)

    # Execute with memory - memory is automatically loaded from Django models
    chain = workflow.to_langchain_chain(
        session_id=session.session_id,
        chat_input="input",
        history="chat_history"
    )

    # Execute the chain
    response = chain.invoke({"input": message})

    # Add AI response to session
    session.add_message(str(response), role="ASSISTANT")

    return {
        "response": str(response),
        "session_id": str(session.session_id),
        "message_count": session.get_message_count()
    }

# Alternative: Using the memory manager directly
def chat_with_direct_memory(session_id, message):
    """Example using memory manager directly."""

    # Get Django-backed memory (default provider)
    memory = get_langchain_memory(session_id, memory_type="buffer")

    # Add user message
    memory.chat_memory.add_user_message(message)

    # Get LLM response (simplified)
    from django_chain.utils.llm_client import create_llm_chat_client
    llm = create_llm_chat_client("openai")

    # Use memory in conversation
    conversation_history = memory.chat_memory.messages
    response = llm.invoke([
        *conversation_history,
        {"role": "user", "content": message}
    ])

    # Add AI response to memory
    memory.chat_memory.add_ai_message(response.content)

    return response.content
```

### **Multi-User Chat Rooms**

```python
class ChatRoomManager:
    """Manage multi-user chat rooms with LLM moderation."""

    def __init__(self):
        self.moderation_workflow = self.create_moderation_workflow()
        self.summary_workflow = self.create_summary_workflow()

    def create_moderation_workflow(self):
        """Create a workflow to moderate chat messages."""

        moderation_prompt = Prompt.objects.create(
            name="chat_moderation",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": """Analyze this chat message for inappropriate content:

Message: {message}
User: {username}
Context: {context}

Return JSON:
{
    "appropriate": true/false,
    "reason": "explanation if inappropriate",
    "severity": "low|medium|high",
    "suggested_action": "none|warn|timeout|ban"
}""",
                "input_variables": ["message", "username", "context"]
            },
            input_variables=["message", "username", "context"]
        )

        return Workflow.objects.create(
            name="chat_moderation",
            prompt=moderation_prompt,
            workflow_definition=[
                {"type": "prompt", "name": "chat_moderation"},
                {"type": "llm", "provider": "openai", "temperature": 0.1},
                {"type": "parser", "parser_type": "json"}
            ]
        )

    def create_summary_workflow(self):
        """Create a workflow to summarize chat conversations."""

        summary_prompt = Prompt.objects.create(
            name="chat_summary",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": """Summarize this chat conversation:

{conversation}

Provide a brief summary of:
1. Main topics discussed
2. Key decisions or conclusions
3. Action items (if any)
4. Overall sentiment

Summary:""",
                "input_variables": ["conversation"]
            },
            input_variables=["conversation"]
        )

        return Workflow.objects.create(
            name="chat_summary",
            prompt=summary_prompt,
            workflow_definition=[
                {"type": "prompt", "name": "chat_summary"},
                {"type": "llm", "provider": "openai"},
                {"type": "parser", "parser_type": "str"}
            ]
        )

    def moderate_message(self, message, user, context=""):
        """Moderate a chat message."""

        chain = self.moderation_workflow.to_langchain_chain()
        result = chain.invoke({
            "message": message,
            "username": user.username,
            "context": context
        })

        return result

    def summarize_conversation(self, session_id, hours=24):
        """Summarize recent conversation in a chat room."""

        since = timezone.now() - timedelta(hours=hours)
        messages = ChatHistory.objects.filter(
            session__session_id=session_id,
            created_at__gte=since
        ).order_by('created_at')

        conversation = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])

        chain = self.summary_workflow.to_langchain_chain()
        summary = chain.invoke({"conversation": conversation})

        return summary
```

## 🔍 **Vector Store and RAG Patterns**

### **Document Processing Pipeline**

```python
from django_chain.services.vector_store_manager import VectorStoreManager
from django_chain.utils.llm_client import create_llm_embedding_client
import os
from pathlib import Path

class DocumentProcessor:
    """Process and index documents for RAG."""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.embedding_client = create_llm_embedding_client()

    def process_text_file(self, file_path, metadata=None):
        """Process a text file and add to vector store."""

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into chunks
        chunks = self.split_text(content, chunk_size=1000, overlap=200)

        # Prepare metadata
        base_metadata = {
            "source": str(file_path),
            "filename": Path(file_path).name,
            "type": "text_file"
        }
        if metadata:
            base_metadata.update(metadata)

        # Add to vector store
        metadatas = [base_metadata.copy() for _ in chunks]
        for i, meta in enumerate(metadatas):
            meta["chunk_id"] = i
            meta["total_chunks"] = len(chunks)

        self.vector_manager.add_documents_to_store(
            texts=chunks,
            metadatas=metadatas,
            embedding_function=self.embedding_client
        )

        return len(chunks)

    def split_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks."""

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

            if i + chunk_size >= len(words):
                break

        return chunks

    def process_directory(self, directory_path, file_patterns=None):
        """Process all files in a directory."""

        if file_patterns is None:
            file_patterns = ["*.txt", "*.md", "*.py", "*.js"]

        directory = Path(directory_path)
        processed_files = 0

        for pattern in file_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    try:
                        chunks = self.process_text_file(
                            file_path,
                            metadata={"category": pattern[2:]}  # Remove *.
                        )
                        processed_files += 1
                        print(f"Processed {file_path}: {chunks} chunks")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        return processed_files

# Usage
processor = DocumentProcessor()
processor.process_directory("/path/to/docs", ["*.md", "*.txt"])
```

### **Advanced RAG with Re-ranking**

```python
class AdvancedRAGSystem:
    """Advanced RAG system with re-ranking and source attribution."""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.embedding_client = create_llm_embedding_client()
        self.rerank_workflow = self.create_rerank_workflow()
        self.rag_workflow = self.create_rag_workflow()

    def create_rerank_workflow(self):
        """Create workflow to re-rank search results."""

        rerank_prompt = Prompt.objects.create(
            name="rerank_documents",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": """Given a query and a list of documents, rank the documents by relevance to the query.

Query: {query}

Documents:
{documents}

Return a JSON list of document indices ordered by relevance (most relevant first):
[2, 0, 1, 3]""",
                "input_variables": ["query", "documents"]
            },
            input_variables=["query", "documents"]
        )

        return Workflow.objects.create(
            name="rerank_documents",
            prompt=rerank_prompt,
            workflow_definition=[
                {"type": "prompt", "name": "rerank_documents"},
                {"type": "llm", "provider": "openai", "temperature": 0.1},
                {"type": "parser", "parser_type": "json"}
            ]
        )

    def create_rag_workflow(self):
        """Create RAG workflow with source attribution."""

        rag_prompt = Prompt.objects.create(
            name="rag_with_sources",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": """Use the following context to answer the question. Include specific source references in your answer.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question using the provided context
2. Include source references like [Source: filename.txt]
3. If the context doesn't contain enough information, say so
4. Be specific and cite relevant sources

Answer:""",
                "input_variables": ["context", "question"]
            },
            input_variables=["context", "question"]
        )

        return Workflow.objects.create(
            name="rag_with_sources",
            prompt=rag_prompt,
            workflow_definition=[
                {"type": "prompt", "name": "rag_with_sources"},
                {"type": "llm", "provider": "openai"},
                {"type": "parser", "parser_type": "str"}
            ]
        )

    def search_and_rerank(self, query, k=10, rerank_top=5):
        """Search documents and re-rank results."""

        # Initial vector search
        results = self.vector_manager.search_documents(
            query=query,
            k=k,
            embedding_function=self.embedding_client
        )

        if not results:
            return []

        # Prepare documents for re-ranking
        documents_text = "\n\n".join([
            f"Document {i}: {result.page_content[:500]}..."
            for i, result in enumerate(results)
        ])

        # Re-rank using LLM
        rerank_chain = self.rerank_workflow.to_langchain_chain()
        rerank_indices = rerank_chain.invoke({
            "query": query,
            "documents": documents_text
        })

        # Reorder results
        reranked_results = []
        for idx in rerank_indices[:rerank_top]:
            if 0 <= idx < len(results):
                reranked_results.append(results[idx])

        return reranked_results

    def answer_with_sources(self, question, k=10):
        """Answer question with source attribution."""

        # Search and re-rank
        results = self.search_and_rerank(question, k=k, rerank_top=5)

        if not results:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": 0.0
            }

        # Prepare context with source information
        context_parts = []
        sources = []

        for i, result in enumerate(results):
            source_info = {
                "filename": result.metadata.get("filename", "unknown"),
                "source": result.metadata.get("source", "unknown"),
                "chunk_id": result.metadata.get("chunk_id", 0)
            }
            sources.append(source_info)

            context_parts.append(
                f"[Source: {source_info['filename']}]\n{result.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Generate answer
        rag_chain = self.rag_workflow.to_langchain_chain()
        answer = rag_chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": answer,
            "sources": sources,
            "confidence": len(results) / k  # Simple confidence score
        }

# Usage
rag_system = AdvancedRAGSystem()
result = rag_system.answer_with_sources("How do I configure Django settings?")

print(f"Answer: {result['answer']}")
print(f"Sources: {[s['filename'] for s in result['sources']]}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 📊 **Analytics and Monitoring Patterns**

### **Custom Analytics Dashboard**

```python
from django.db.models import Count, Avg, Sum, Q
from django.utils import timezone
from datetime import timedelta
import json

class LLMAnalytics:
    """Comprehensive LLM usage analytics."""

    def get_usage_metrics(self, days=30):
        """Get comprehensive usage metrics."""

        since = timezone.now() - timedelta(days=days)
        logs = InteractionLog.objects.filter(created_at__gte=since)

        metrics = {
            "overview": self._get_overview_metrics(logs),
            "performance": self._get_performance_metrics(logs),
            "costs": self._get_cost_metrics(logs),
            "errors": self._get_error_metrics(logs),
            "workflows": self._get_workflow_metrics(logs),
            "users": self._get_user_metrics(logs)
        }

        return metrics

    def _get_overview_metrics(self, logs):
        """Get overview metrics."""

        total = logs.count()
        successful = logs.filter(status="SUCCESS").count()
        failed = logs.filter(status="FAILURE").count()

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "total_tokens": logs.aggregate(
                total=Sum('input_tokens') + Sum('output_tokens')
            )['total'] or 0
        }

    def _get_performance_metrics(self, logs):
        """Get performance metrics."""

        return {
            "average_latency": logs.aggregate(Avg('latency_ms'))['latency_ms__avg'] or 0,
            "p95_latency": self._percentile(logs, 'latency_ms', 95),
            "p99_latency": self._percentile(logs, 'latency_ms', 99),
            "requests_per_hour": self._get_requests_per_hour(logs)
        }

    def _get_cost_metrics(self, logs):
        """Calculate cost metrics."""

        # OpenAI pricing (update as needed)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0}
        }

        total_cost = 0
        model_costs = {}

        for log in logs:
            model = log.model_name
            if model in pricing:
                input_cost = (log.input_tokens / 1000) * pricing[model]["input"]
                output_cost = (log.output_tokens / 1000) * pricing[model]["output"]
                cost = input_cost + output_cost

                total_cost += cost
                model_costs[model] = model_costs.get(model, 0) + cost

        return {
            "total_cost": total_cost,
            "cost_by_model": model_costs,
            "average_cost_per_request": total_cost / logs.count() if logs.count() > 0 else 0
        }

    def _get_error_metrics(self, logs):
        """Get error analysis."""

        error_logs = logs.filter(status="FAILURE")

        error_types = error_logs.values('error_message').annotate(
            count=Count('id')
        ).order_by('-count')[:10]

        return {
            "total_errors": error_logs.count(),
            "error_rate": (error_logs.count() / logs.count() * 100) if logs.count() > 0 else 0,
            "top_error_types": list(error_types)
        }

    def _get_workflow_metrics(self, logs):
        """Get workflow usage metrics."""

        workflow_stats = logs.values('workflow__name').annotate(
            count=Count('id'),
            avg_latency=Avg('latency_ms'),
            success_rate=Count('id', filter=Q(status="SUCCESS")) * 100.0 / Count('id')
        ).order_by('-count')

        return list(workflow_stats)

    def _get_user_metrics(self, logs):
        """Get user usage metrics."""

        user_stats = logs.filter(user__isnull=False).values('user__username').annotate(
            count=Count('id'),
            total_tokens=Sum('input_tokens') + Sum('output_tokens')
        ).order_by('-count')[:20]

        return list(user_stats)

    def _percentile(self, queryset, field, percentile):
        """Calculate percentile for a field."""

        values = list(queryset.values_list(field, flat=True))
        if not values:
            return 0

        values.sort()
        k = (len(values) - 1) * percentile / 100
        f = int(k)
        c = k - f

        if f + 1 < len(values):
            return values[f] * (1 - c) + values[f + 1] * c
        else:
            return values[f]

    def _get_requests_per_hour(self, logs):
        """Get requests per hour over time."""

        # Group by hour
        from django.db.models import DateTimeField
        from django.db.models.functions import TruncHour

        hourly_stats = logs.annotate(
            hour=TruncHour('created_at')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')

        return list(hourly_stats)

# Usage in a view
def analytics_api(request):
    """API endpoint for analytics data."""

    days = int(request.GET.get('days', 7))
    analytics = LLMAnalytics()
    metrics = analytics.get_usage_metrics(days=days)

    return JsonResponse(metrics)
```

### **Real-time Monitoring**

```python
from django.core.cache import cache
from django.utils import timezone
import json

class RealTimeMonitor:
    """Real-time monitoring for LLM operations."""

    def __init__(self):
        self.cache_timeout = 300  # 5 minutes

    def track_request(self, workflow_name, latency_ms, tokens, status):
        """Track a request in real-time."""

        now = timezone.now()
        minute_key = now.strftime("%Y%m%d%H%M")

        # Update minute-level metrics
        metrics_key = f"llm_metrics_{minute_key}"
        metrics = cache.get(metrics_key, {
            "requests": 0,
            "total_latency": 0,
            "total_tokens": 0,
            "errors": 0,
            "workflows": {}
        })

        metrics["requests"] += 1
        metrics["total_latency"] += latency_ms
        metrics["total_tokens"] += tokens

        if status == "FAILURE":
            metrics["errors"] += 1

        # Track workflow-specific metrics
        if workflow_name not in metrics["workflows"]:
            metrics["workflows"][workflow_name] = {
                "requests": 0,
                "total_latency": 0,
                "errors": 0
            }

        workflow_metrics = metrics["workflows"][workflow_name]
        workflow_metrics["requests"] += 1
        workflow_metrics["total_latency"] += latency_ms

        if status == "FAILURE":
            workflow_metrics["errors"] += 1

        cache.set(metrics_key, metrics, self.cache_timeout)

        # Update current metrics
        self._update_current_metrics(metrics)

    def _update_current_metrics(self, metrics):
        """Update current real-time metrics."""

        current = {
            "requests_per_minute": metrics["requests"],
            "average_latency": metrics["total_latency"] / metrics["requests"] if metrics["requests"] > 0 else 0,
            "tokens_per_minute": metrics["total_tokens"],
            "error_rate": (metrics["errors"] / metrics["requests"] * 100) if metrics["requests"] > 0 else 0,
            "top_workflows": sorted(
                metrics["workflows"].items(),
                key=lambda x: x[1]["requests"],
                reverse=True
            )[:5]
        }

        cache.set("llm_current_metrics", current, 60)  # 1 minute cache

    def get_current_metrics(self):
        """Get current real-time metrics."""

        return cache.get("llm_current_metrics", {
            "requests_per_minute": 0,
            "average_latency": 0,
            "tokens_per_minute": 0,
            "error_rate": 0,
            "top_workflows": []
        })

    def get_recent_activity(self, minutes=60):
        """Get recent activity over the last N minutes."""

        now = timezone.now()
        activity = []

        for i in range(minutes):
            minute = now - timedelta(minutes=i)
            minute_key = minute.strftime("%Y%m%d%H%M")
            metrics_key = f"llm_metrics_{minute_key}"

            metrics = cache.get(metrics_key, {
                "requests": 0,
                "total_latency": 0,
                "errors": 0
            })

            activity.append({
                "timestamp": minute.isoformat(),
                "requests": metrics["requests"],
                "average_latency": metrics["total_latency"] / metrics["requests"] if metrics["requests"] > 0 else 0,
                "errors": metrics["errors"]
            })

        return list(reversed(activity))

# Integration with LoggingHandler
class EnhancedLoggingHandler(LoggingHandler):
    """Enhanced logging handler with real-time monitoring."""

    def __init__(self):
        super().__init__()
        self.monitor = RealTimeMonitor()

    def on_llm_end(self, response, **kwargs):
        """Enhanced LLM end callback with monitoring."""

        # Call parent method
        super().on_llm_end(response, **kwargs)

        # Track in real-time monitor
        workflow_name = kwargs.get('workflow_name', 'unknown')
        latency_ms = kwargs.get('latency_ms', 0)
        tokens = kwargs.get('total_tokens', 0)

        self.monitor.track_request(
            workflow_name=workflow_name,
            latency_ms=latency_ms,
            tokens=tokens,
            status="SUCCESS"
        )

    def on_llm_error(self, error, **kwargs):
        """Enhanced LLM error callback with monitoring."""

        # Call parent method
        super().on_llm_error(error, **kwargs)

        # Track error in real-time monitor
        workflow_name = kwargs.get('workflow_name', 'unknown')

        self.monitor.track_request(
            workflow_name=workflow_name,
            latency_ms=0,
            tokens=0,
            status="FAILURE"
        )
```

## 🎯 **Next Steps**

These examples demonstrate common patterns and advanced use cases for Django Chain. To continue learning:

1. **Study the [Vanilla Django Example](https://github.com/Brian-Kariu/django-chain/tree/main/examples/vanilla_django)** for complete implementations
2. **Read the [How-to Guides](how-to-guides.md)** for specific problem-solving approaches
3. **Explore [Advanced Topics](../advanced/custom-providers.md)** for customization
4. **Check the [API Reference](../api/intro.md)** for detailed documentation

## 💡 **Best Practices**

- **Use descriptive names** for prompts and workflows
- **Version your prompts** to track changes
- **Monitor token usage** to control costs
- **Implement proper error handling** for production use
- **Test with fake providers** before using real LLMs
- **Cache responses** where appropriate to reduce costs
- **Sanitize sensitive data** in logs and outputs

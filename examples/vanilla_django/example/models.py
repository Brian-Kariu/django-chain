"""
Example models demonstrating django-chain integration patterns.

This module showcases:
- Custom models that work with django-chain
- Document storage for vector operations
- Integration with django-chain models
- Best practices for model design
"""

import uuid

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

from django_chain.models import Workflow, ChatSession

User = get_user_model()


class ExampleDocument(models.Model):
    """
    Example document model for demonstrating vector store operations.

    This shows how developers can create their own document models
    that integrate with django-chain's vector store functionality.
    """

    title = models.CharField(max_length=255)
    content = models.TextField()
    author = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="User who created this document",
    )
    tags = models.JSONField(default=list, blank=True, help_text="Tags for categorizing documents")
    metadata = models.JSONField(
        default=dict, blank=True, help_text="Additional metadata for the document"
    )
    is_indexed = models.BooleanField(
        default=False, help_text="Whether this document has been added to vector store"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["is_indexed", "created_at"]),
            models.Index(fields=["author", "created_at"]),
        ]

    def __str__(self) -> str:
        return self.title

    def mark_as_indexed(self):
        """Mark this document as indexed in the vector store."""
        self.is_indexed = True
        self.save(update_fields=["is_indexed", "updated_at"])


class ExampleWorkflowExecution(models.Model):
    """
    Example model for tracking custom workflow executions.

    Demonstrates how developers can create their own tracking models
    that complement django-chain's InteractionLog.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(
        Workflow, on_delete=models.CASCADE, help_text="The workflow that was executed"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="User who executed the workflow",
    )
    input_data = models.JSONField(help_text="Input data provided to the workflow")
    output_data = models.JSONField(
        null=True, blank=True, help_text="Output data from the workflow execution"
    )
    execution_time_ms = models.IntegerField(
        null=True, blank=True, help_text="Time taken to execute the workflow in milliseconds"
    )
    success = models.BooleanField(default=False, help_text="Whether the execution was successful")
    error_message = models.TextField(
        null=True, blank=True, help_text="Error message if execution failed"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["workflow", "created_at"]),
            models.Index(fields=["user", "success"]),
            models.Index(fields=["success", "created_at"]),
        ]

    def __str__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"{self.workflow.name} - {status} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


class ExampleChatAnalytics(models.Model):
    """
    Example model for chat analytics and insights.

    Shows how developers can build analytics on top of django-chain's
    chat functionality.
    """

    chat_session = models.OneToOneField(
        ChatSession, on_delete=models.CASCADE, related_name="analytics"
    )
    total_messages = models.IntegerField(default=0)
    total_user_messages = models.IntegerField(default=0)
    total_assistant_messages = models.IntegerField(default=0)
    average_response_time_ms = models.FloatField(
        null=True, blank=True, help_text="Average response time for assistant messages"
    )
    total_tokens_used = models.IntegerField(
        default=0, help_text="Total tokens used in this session"
    )
    session_duration_minutes = models.IntegerField(
        null=True, blank=True, help_text="Duration of the chat session in minutes"
    )
    last_activity = models.DateTimeField(
        null=True, blank=True, help_text="Last activity in this chat session"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"Analytics for {self.chat_session.title or self.chat_session.session_id}"

    def update_analytics(self):
        """Update analytics based on current chat session data."""
        messages = self.chat_session.messages.filter(is_deleted=False)

        self.total_messages = messages.count()
        self.total_user_messages = messages.filter(role="USER").count()
        self.total_assistant_messages = messages.filter(role="ASSISTANT").count()

        # Calculate session duration
        if messages.exists():
            first_message = messages.order_by("timestamp").first()
            last_message = messages.order_by("timestamp").last()
            duration = last_message.timestamp - first_message.timestamp
            self.session_duration_minutes = int(duration.total_seconds() / 60)
            self.last_activity = last_message.timestamp

        # Calculate total tokens
        self.total_tokens_used = sum(msg.token_count for msg in messages if msg.token_count)

        self.save()


class ExamplePromptTemplate(models.Model):
    """
    Example of how developers might extend prompt functionality.

    This model shows patterns for creating domain-specific prompt templates
    that work alongside django-chain's Prompt model.
    """

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    category = models.CharField(
        max_length=100,
        choices=[
            ("customer_service", "Customer Service"),
            ("content_generation", "Content Generation"),
            ("analysis", "Analysis"),
            ("translation", "Translation"),
            ("coding", "Coding Assistant"),
        ],
        default="customer_service",
    )
    template_text = models.TextField(
        help_text="The prompt template with {variables} for substitution"
    )
    required_variables = models.JSONField(default=list, help_text="List of required variable names")
    optional_variables = models.JSONField(default=list, help_text="List of optional variable names")
    example_input = models.JSONField(
        default=dict, blank=True, help_text="Example input data for testing"
    )
    usage_count = models.IntegerField(
        default=0, help_text="Number of times this template has been used"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-usage_count", "name"]
        indexes = [
            models.Index(fields=["category", "is_active"]),
            models.Index(fields=["usage_count"]),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.category})"

    def increment_usage(self):
        """Increment the usage count for this template."""
        self.usage_count += 1
        self.save(update_fields=["usage_count", "updated_at"])

    def to_django_chain_format(self):
        """Convert this template to django-chain Prompt format."""
        return {
            "langchain_type": "PromptTemplate",
            "template": self.template_text,
            "input_variables": self.required_variables,
        }


# Legacy models kept for backward compatibility
class TestChain(models.Model):
    """Legacy test model - kept for backward compatibility."""

    name = models.CharField(max_length=100)
    chain = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


class TestSession(models.Model):
    """Legacy test model - kept for backward compatibility."""

    name = models.CharField(max_length=100)
    session = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

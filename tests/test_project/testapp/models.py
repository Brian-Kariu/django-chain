"""
Test models for integration testing.
"""

import uuid
from django.db import models
from django_chain.models import OutputParsers, Prompt


class TestPrompt(Prompt):
    created_at = models.DateTimeField(auto_now_add=True)


class Joke(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    setup = models.CharField(
        max_length=255, null=False, blank=False, help_text="Question to setup joke"
    )
    punchline = models.CharField(
        max_length=255, null=False, blank=False, help_text="Answer to resolve the joke"
    )


class TestOutputParser(OutputParsers):
    output_model = models.OneToOneField(Joke, on_delete=models.CASCADE, null=False, blank=False)


class TestDocument(models.Model):
    """Test model for document operations."""

    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class TestChain(models.Model):
    """Test model for chain operations."""

    name = models.CharField(max_length=100)
    chain = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


class TestSession(models.Model):
    """Test model for session operations."""

    name = models.CharField(max_length=100)
    session = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

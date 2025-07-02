"""
Test models for integration testing.
"""

import uuid

from django.db import models


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

"""Common models shared across the project"""

import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _


class AbstractBase(models.Model):
    """Abstract class for audit purposes."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    created = models.DateTimeField(auto_now_add=True, db_index=True)
    created_by = models.UUIDField(null=True, blank=True, editable=False)
    updated = models.DateTimeField(auto_now=True, db_index=True)
    updated_by = models.UUIDField(null=True, blank=True)
    active = models.BooleanField(default=True)

    class Meta:
        abstract = True
        ordering: tuple[str, ...] = ("-updated", "-created")

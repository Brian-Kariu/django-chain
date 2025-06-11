"""
Chat models for django-chain.
"""

from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class ChatSession(models.Model):
    """
    A model for storing chat session information.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        help_text=_("User associated with this chat session"),
    )
    session_id = models.CharField(
        _("session id"),
        max_length=100,
        unique=True,
        help_text=_("UUID for anonymous sessions or custom session tracking"),
    )
    title = models.CharField(
        _("title"),
        max_length=200,
        blank=True,
        null=True,
        help_text=_("A user-friendly title for the chat"),
    )
    llm_config = models.JSONField(
        _("LLM config"),
        default=dict,
        help_text=_("Specific LLM configuration for this session"),
    )
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)
    updated_at = models.DateTimeField(_("updated at"), auto_now=True)

    class Meta:
        verbose_name = _("Chat Session")
        verbose_name_plural = _("Chat Sessions")
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
        ]

    def __str__(self) -> str:
        return self.title or f"Chat Session {self.session_id}"


class ChatMessage(models.Model):
    """
    A model for storing chat messages.
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    content = models.TextField(_("content"))
    role = models.CharField(
        _("role"),
        max_length=10,
        choices=ROLE_CHOICES,
    )
    timestamp = models.DateTimeField(_("timestamp"), auto_now_add=True)
    token_count = models.IntegerField(
        _("token count"),
        null=True,
        blank=True,
    )
    order = models.IntegerField(
        _("order"),
        default=0,
        help_text=_("For ordering in case of simultaneous writes"),
    )

    class Meta:
        verbose_name = _("Chat Message")
        verbose_name_plural = _("Chat Messages")
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["session", "timestamp"]),
        ]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}..."

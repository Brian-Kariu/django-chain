"""
Log models for django-chain.
"""

from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class LLMInteractionLog(models.Model):
    """
    A model for logging LLM interactions.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("User who initiated the interaction"),
    )
    prompt_text = models.TextField(_("prompt text"))
    response_text = models.TextField(_("response text"))
    model_name = models.CharField(
        _("model name"),
        max_length=100,
        help_text=_("Name of the LLM model used"),
    )
    provider = models.CharField(
        _("provider"),
        max_length=50,
        help_text=_("LLM provider (e.g., openai, google)"),
    )
    input_tokens = models.IntegerField(_("input tokens"), null=True, blank=True)
    output_tokens = models.IntegerField(_("output tokens"), null=True, blank=True)
    total_cost = models.DecimalField(
        _("total cost"),
        max_digits=10,
        decimal_places=8,
        null=True,
        blank=True,
        help_text=_("Estimated cost of the interaction in USD"),
    )
    latency_ms = models.IntegerField(
        _("latency ms"),
        null=True,
        blank=True,
        help_text=_("Latency of the LLM API call in milliseconds"),
    )
    status = models.CharField(
        _("status"),
        max_length=20,
        choices=[("success", "Success"), ("error", "Error")],
        default="success",
    )
    error_message = models.TextField(
        _("error message"),
        blank=True,
        null=True,
        help_text=_("Error message if the interaction failed"),
    )
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)
    metadata = models.JSONField(
        _("metadata"),
        default=dict,
        blank=True,
        help_text=_("Additional arbitrary metadata"),
    )

    class Meta:
        verbose_name = _("LLM Interaction Log")
        verbose_name_plural = _("LLM Interaction Logs")
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"Log {self.pk} - {self.model_name} ({self.status})"

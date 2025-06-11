"""
Admin interface for django-chain.
"""

from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from django_chain.models.chains import LLMChain
from django_chain.models.chat import ChatMessage, ChatSession
from django_chain.models.logs import LLMInteractionLog


@admin.register(LLMChain)
class LLMChainAdmin(admin.ModelAdmin):
    """Admin interface for LLMChain model."""

    list_display = (
        "name",
        "provider",
        "model_name",
        "is_active",
        "created_at",
        "updated_at",
    )
    list_filter = ("provider", "is_active", "created_at")
    search_fields = ("name", "description", "prompt_template")
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        (None, {"fields": ("name", "description", "is_active")}),
        ("LLM Configuration", {"fields": ("provider", "model_name", "temperature")}),
        (
            "Prompt Configuration",
            {"fields": ("prompt_template", "input_variables", "output_parser")},
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    """Admin interface for ChatSession model."""

    list_display = ("session_id", "title", "user", "created_at", "updated_at")
    list_filter = ("created_at", "updated_at")
    search_fields = ("session_id", "title")
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        (None, {"fields": ("session_id", "title", "user")}),
        ("LLM Configuration", {"fields": ("llm_config",)}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    """Admin interface for ChatMessage model."""

    list_display = ("session", "role", "content_preview", "timestamp", "token_count")
    list_filter = ("role", "timestamp")
    search_fields = ("content", "session__session_id")
    readonly_fields = ("timestamp",)
    fieldsets = (
        (None, {"fields": ("session", "role", "content")}),
        ("Timestamps", {"fields": ("timestamp",), "classes": ("collapse",)}),
    )

    def get_session_title(self, obj):
        """Return a link to the session's admin page."""
        url = reverse("admin:django_chain_chatsession_change", args=[obj.session.id])
        return format_html('<a href="{}">{}</a>', url, obj.session.title)

    get_session_title.short_description = "Session"
    get_session_title.admin_order_field = "session__title"

    def content_preview(self, obj: ChatMessage) -> str:
        """Return a preview of the message content."""
        return format_html(
            '<span title="{}">{}</span>',
            obj.content,
            obj.content[:50] + "..." if len(obj.content) > 50 else obj.content,
        )

    content_preview.short_description = "Content"
    content_preview.admin_order_field = "content"


@admin.register(LLMInteractionLog)
class LLMInteractionLogAdmin(admin.ModelAdmin):
    """Admin interface for LLMInteractionLog model."""

    list_display = (
        "model_name",
        "input_tokens",
        "output_tokens",
        "total_cost",
        "status",
        "created_at",
    )
    list_filter = ("model_name", "status", "created_at")
    search_fields = ("prompt_text", "response_text", "user__username")
    readonly_fields = ("created_at",)
    fieldsets = (
        (None, {"fields": ("user", "model_name", "status")}),
        ("Interaction Details", {"fields": ("prompt_text", "response_text")}),
        (
            "Metrics",
            {"fields": ("input_tokens", "output_tokens", "cost", "latency_ms")},
        ),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )

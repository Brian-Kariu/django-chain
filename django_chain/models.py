"""
Models for django-chain: LLM prompts, workflows, chat sessions, messages, logs, and user interactions.

This module defines the core database models for prompt management, workflow orchestration, chat memory,
LLM interaction logging, and user interaction tracking in Django Chain.

Typical usage example:
    prompt = Prompt.objects.create(...)
    session = ChatSession.objects.create(...)
    message = ChatMessage.objects.create(session=session, ...)

Raises:
    ValidationError: If model constraints are violated.
"""

import logging
import uuid
from functools import reduce
from typing import Any
from typing import Optional

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db import transaction
from django.db.models import ForeignKey
from django.db.models import Max
from django.forms.models import model_to_dict
from django.utils.translation import gettext_lazy as _
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from django_chain.exceptions import PromptValidationError
from django_chain.exceptions import WorkflowValidationError
from django_chain.utils.langchain import Workflow as WorkflowProcessor
from django_chain.utils.llm_client import LoggingHandler
from django_chain.utils.llm_client import _sanitize_model_parameters
from django_chain.utils.llm_client import _sanitize_sensitive_data
from django_chain.utils.llm_client import add_wrapper_function
from django_chain.utils.memory_manager import get_chat_history
from django_chain.utils.memory_manager import get_langchain_memory
from django_chain.utils.prompts import _convert_to_prompt_template
from django_chain.validators import validate_prompt
from django_chain.validators import validate_workflow

# Initialize logger at module level
logger = logging.getLogger(__name__)


class AIRequestStatus(models.TextChoices):
    """
    Enum for chat message roles.
    """

    PROCESSING = "PROCESSING", _("Request is processing")
    SUCCESS = "SUCCESS", _("Request has succeeded")
    FAILURE = "FAILURE", _("Request has failed")


class RoleChoices(models.TextChoices):
    """
    Enum for chat message roles.
    """

    USER = "USER", _("User template")
    ASSISTANT = "ASSISTANT", _("Assistant Template")
    SYSTEM = "SYSTEM", _("System Template")


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        abstract = True


class AuditModel(TimeStampedModel):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    created_by = models.UUIDField(null=True, blank=True, editable=False)
    updated_by = models.UUIDField(null=True, blank=True)

    class Meta:
        abstract = True
        ordering: tuple[str, ...] = ("-updated_at", "-created_at")


class VersionedManager(models.Manager):
    def get_active(self, name):
        """
        Returns the active object for a given name.
        """
        return self.filter(name=name, is_active=True).first()

    def deactivate_all_for_name(self, name, exclude_pk=None):
        """
        Deactivates all active objects for a given name,
        optionally excluding a specific primary key.
        """
        qs = self.filter(name=name, is_active=True)
        if exclude_pk:
            qs = qs.exclude(pk=exclude_pk)
        return qs.update(is_active=False)


class VersionedModel(models.Model):
    name = models.CharField(max_length=255, unique=True, null=False)
    version = models.PositiveIntegerField(default=1, null=False, blank=False)
    is_active = models.BooleanField(
        default=False,
        help_text=_(
            "Activates/Deactivates the object, only one object of a given 'name' should be active at a given time"
        ),
    )
    objects = VersionedManager()

    class Meta:
        abstract = True
        unique_together = (("name", "version"), ("name", "is_active"))
        ordering = ["name", "-version"]

    def __str__(self) -> str:
        return f"{self.name} ({'Active' if self.is_active else 'Inactive'})"

    def clean(self):
        super().clean()
        if self.is_active:
            active_objects = self.__class__.objects.filter(name=self.name, is_active=True)
            if self.pk:
                active_objects = active_objects.exclude(pk=self.pk)
            if active_objects.exists():
                raise ValidationError(
                    _(
                        "There can only be one active prompt per name. "
                        "Please deactivate the existing active prompt before setting this one as active."
                    ),
                    code="duplicate_active_prompt",
                )

    def save(self, *args, **kwargs):
        self.full_clean()
        if self.is_active:
            with transaction.atomic():
                self.__class__.objects.deactivate_all_for_name(self.name, exclude_pk=self.pk)
                super().save(*args, **kwargs)
        super().save(*args, **kwargs)

    def activate(self):
        if not self.is_active:
            self.is_active = True
            self.save()

    def deactivate(self):
        if self.is_active:
            self.is_active = False
            self.save()

    @classmethod
    def create_new_version(cls, name: str, activate: bool = True, **model_specific_data):
        """
        Creates a new version of a VersionedModel instance.

        Args:
            name (str): The name of the versioned object.
            activate (bool): Whether this new version should be set as active.
                             If True, all other active versions with the same name will be deactivated.
            **model_specific_data: Keyword arguments for fields specific to the concrete model
                                   (e.g., 'template_string', 'description' for PromptTemplate).

        Returns:
            The newly created concrete model instance.
        """
        max_version = cls.objects.filter(name=name).aggregate(Max("version"))["version__max"]
        new_version_number = (max_version or 0) + 1

        new_object = cls(
            name=name,
            version=new_version_number,
            is_active=activate,
            **model_specific_data,
        )

        new_object.save()
        return new_object


class AbstractPrompt(VersionedModel):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    name = models.CharField(max_length=255, unique=True, null=False)
    prompt_template = models.JSONField(
        default=dict,
        null=False,
        blank=False,
        help_text=_(
            "JSON representation of the LangChain prompt. Must include 'langchain_type' (e.g., 'PromptTemplate', 'ChatPromptTemplate')."
        ),
        validators=[validate_prompt],
    )
    input_variables = models.JSONField(
        help_text="Input variables to the prompt", blank=True, null=True
    )
    optional_variables = models.JSONField(
        help_text="Input variables to the prompt", blank=True, null=True
    )

    class Meta:
        abstract = True
        ordering: tuple[str, ...] = ("-updated_at", "-created_at", "name", "-version")

    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({'Active' if self.is_active else 'Inactive'})"

    def clean(self):
        super().clean()
        converted_langchain_object = self.to_langchain_prompt()
        if not isinstance(self.prompt_template, dict):
            raise ValidationError(
                _("Prompt template must be a JSON object."), code="invalid_prompt_template_format"
            )
        if "langchain_type" not in self.prompt_template:
            raise ValidationError(
                _(
                    "Prompt template JSON must contain a 'langchain_type' key (e.g., 'PromptTemplate', 'ChatPromptTemplate')."
                ),
                code="missing_langchain_type",
            )

        if not converted_langchain_object:
            msg = "Prompt template submitted cannot generate a valid langchain prompt template."
            raise PromptValidationError(value=self.prompt_template, additional_message=msg)

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def get_prompt_content(self):
        return self.prompt_template

    def get_input_variables(self):
        return self.input_variables if self.input_variables is not None else []

    def to_langchain_prompt(self):  # noqa: C901
        """
        Converts the stored JSON prompt_template into an actual LangChain prompt object.
        """
        return _convert_to_prompt_template(self)


class Prompt(AuditModel, AbstractPrompt):
    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the Workflow instance for API responses.

        Returns:
            dict: Dictionary with workflow fields.
        """
        data = model_to_dict(self, exclude=["id"])
        formatted_data = {"id": str(self.id), **data}
        return formatted_data

    class Meta:
        verbose_name = _("Prompt")
        verbose_name_plural = _("Prompts")
        ordering: tuple[str, ...] = ("-updated_at", "-created_at", "name", "-version")


class AbstractWorkflow(VersionedModel):
    """
    Represents an AI workflow, defined as a sequence of LangChain components.

    Attributes:
        id (UUID): Unique identifier for the workflow.
        name (str): Name of the workflow (unique).
        description (str): Description of the workflow.
        workflow_definition (list): List of steps (dicts) defining the workflow. is_active (bool): Whether this workflow is active.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(
        max_length=255,
        unique=True,
        help_text=_(
            "A unique name for this workflow (e.g., 'SummaryGenerator', 'CustomerServiceChatbot')."
        ),
    )
    description = models.TextField(blank=True, null=True)
    prompt = ForeignKey(Prompt, on_delete=models.SET_NULL, null=True, blank=True)
    workflow_definition = models.JSONField(
        default=dict,
        null=False,
        blank=False,
        help_text=_(
            "JSON array defining the sequence of LangChain components (prompt, llm, parser)."
        ),
        validators=[validate_workflow],
    )

    class Meta:
        abstract = True
        verbose_name = _("Workflow")
        verbose_name_plural = _("Workflows")
        ordering: tuple[str, ...] = ("-updated_at", "-created_at", "name", "-version")

    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({'Active' if self.is_active else 'Inactive'})"

    def to_langchain_chain(self, *args, **kwargs) -> Any:  # noqa C901
        """
        Convert this workflow instance to a LangChain chain.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including:
                - log: Whether to enable logging ("true"/"false")
                - session_id: Session ID for chat history
                - llm_config: LLM configuration dictionary
                - chat_input: Chat input key for message history
                - history: History key for message history

        Returns:
            LangChain chain object ready for execution
        """
        workflow_processor = WorkflowProcessor()

        config_override = {"prompt_instance": self.prompt}
        current_config = {
            **kwargs,
            **config_override,
        }

        chain_components = workflow_processor.convert_to_runnable_components(
            self.workflow_definition, **current_config
        )

        logging_toggle = kwargs.get("log")
        if logging_toggle == "true":
            user = kwargs.get("user")

            if user and hasattr(user, "is_authenticated") and user.is_authenticated:
                user_to_assign = user
            else:
                user_to_assign = None

            llm_config = kwargs.get("llm_config", {})
            default_chat_model = llm_config.get("DEFAULT_CHAT_MODEL", {})

            sanitized_model_params = _sanitize_model_parameters(default_chat_model)

            interaction_log = InteractionLog.objects.create(
                workflow=self,
                user=user_to_assign,
                model_name=default_chat_model.get("name", "unknown"),
                provider=llm_config.get("DEFAULT_LLM_PROVIDER", "unknown"),
                model_parameters=sanitized_model_params,
                status="PROCESSING",
            )
            workflow_chain = reduce(lambda a, b: a | b, chain_components).with_config(
                callbacks=interaction_log.get_logging_handler(handler="basic")
            )
        else:
            workflow_chain = reduce(lambda a, b: a | b, chain_components)

        if kwargs.get("session_id"):
            input_messages_key = kwargs.get("chat_input")
            history = kwargs.get("history")
            return add_wrapper_function(
                chain=workflow_chain,
                function_name="runnable_with_message_history",
                input_messages_key=input_messages_key,
                history_messages_key=history,
            )
        else:
            return workflow_chain


class Workflow(AuditModel, AbstractWorkflow):
    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the Workflow instance for API responses.

        Returns:
            dict: Dictionary with workflow fields.
        """
        data = model_to_dict(self, exclude=["id"])
        formatted_data = {"id": str(self.id), **data}
        return formatted_data

    class Meta:
        verbose_name = _("Workflow")
        verbose_name_plural = _("Workflows")
        ordering: tuple[str, ...] = ("-updated_at", "-created_at", "name", "-version")


class ChatSessionManager(models.Manager):
    """
    Manager for ChatSession model, providing helper methods for active/inactive sessions.
    """

    def active(self):
        """Return queryset of active chat sessions."""
        return self.filter(is_active=True)

    def archived(self):
        """Return queryset of archived (inactive) chat sessions."""
        return self.filter(is_active=False)

    def for_user(self, user):
        """Return queryset of sessions for a specific user."""
        return self.filter(user=user)

    def active_for_user(self, user):
        """Return queryset of active sessions for a specific user."""
        return self.filter(user=user, is_active=True)


class ChatHistoryManager(models.Manager):
    """
    Manager for ChatHistory model, providing helper methods for non-deleted messages.
    """

    def active(self):
        """Return queryset of non-deleted chat messages."""
        return self.filter(is_deleted=False)

    def deleted(self):
        """Return queryset of soft-deleted chat messages."""
        return self.filter(is_deleted=True)

    def for_session(self, session):
        """Return queryset of messages for a specific session (non-deleted only)."""
        return self.filter(session=session, is_deleted=False)

    def all_for_session(self, session):
        """Return queryset of all messages for a specific session (including deleted)."""
        return self.filter(session=session)


class AbstractChatSession(TimeStampedModel):
    """
    Stores chat session information, including user, session ID, and LLM config.

    Attributes:
        user (User): Associated user (nullable).
        session_id (UUID): Unique session identifier (auto-generated UUID).
        title (str): Optional title for the chat session.
        workflow (Workflow): Associated workflow for this session.
        is_active (bool): Whether the session is currently active.
        created_at (datetime): Creation timestamp.
        updated_at (datetime): Last update timestamp.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        help_text="User associated with this chat session",
    )
    session_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        help_text="Unique UUID identifier for this chat session",
    )
    title = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        help_text="A user-friendly title for the chat",
    )
    workflow = ForeignKey(AbstractWorkflow, on_delete=models.PROTECT, null=True, blank=True)
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the session is currently active (for archiving)",
        db_index=True,
    )

    objects = ChatSessionManager()

    class Meta:
        abstract = True
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"
        ordering: tuple[str, ...] = ("-updated_at", "-created_at")
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
            models.Index(fields=["is_active", "user"]),  # For filtering active sessions by user
            models.Index(fields=["is_active", "created_at"]),  # For listing active sessions
        ]

    def __str__(self) -> str:
        return self.title or f"Chat Session {self.session_id}"

    def archive(self) -> None:
        """Archive this chat session by setting is_active to False."""
        self.is_active = False
        self.save(update_fields=["is_active", "updated_at"])

    def activate(self) -> None:
        """Activate this chat session by setting is_active to True."""
        self.is_active = True
        self.save(update_fields=["is_active", "updated_at"])


class ChatSession(AbstractChatSession):
    workflow = ForeignKey(Workflow, on_delete=models.PROTECT, null=True, blank=True)

    class Meta:
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"
        ordering: tuple[str, ...] = ("-updated_at", "-created_at")
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
            models.Index(fields=["is_active", "user"]),  # For filtering active sessions by user
            models.Index(fields=["is_active", "created_at"]),  # For listing active sessions
        ]

    def get_memory(self, memory_type: str = "buffer", k: Optional[int] = None):
        """
        Get a LangChain memory object for this session.

        Args:
            memory_type: Type of memory ('buffer' or 'buffer_window')
            k: Number of messages to keep for window memory

        Returns:
            Configured LangChain memory object
        """
        return get_langchain_memory(
            session_id=self.session_id, memory_type=memory_type, k=k, provider="django"
        )

    def get_chat_history(self):
        """
        Get a LangChain-compatible chat history for this session.

        Returns:
            Chat message history instance
        """
        return get_chat_history(self.session_id, provider="django")

    def add_message(self, content: str, role: str = "USER", **kwargs):
        """
        Add a message to this chat session.

        Args:
            content: Message content
            role: Message role (USER, ASSISTANT, SYSTEM)
            **kwargs: Additional fields for the ChatHistory model

        Returns:
            Created ChatHistory instance
        """
        # Get the next order number
        last_message = self.messages.order_by("-order").first()
        next_order = (last_message.order + 1) if last_message else 0

        return ChatHistory.objects.create(
            session=self, content=content, role=role, order=next_order, **kwargs
        )

    def clear_messages(self, hard_delete: bool = False):
        """
        Clear all messages from this session.

        Args:
            hard_delete: If True, permanently delete messages.
                        If False, soft delete (set is_deleted=True)
        """
        if hard_delete:
            with transaction.atomic():
                messages_to_delete = self.messages.all().select_for_update()
                deleted_count = messages_to_delete.delete()[0]
                logger.warning(
                    f"Permanently deleted {deleted_count} messages from session {self.session_id}. "
                    "This action cannot be undone."
                )
        else:
            updated_count = self.messages.filter(is_deleted=False).update(is_deleted=True)
            logger.info(f"Soft deleted {updated_count} messages from session {self.session_id}")

    def get_message_count(self, include_deleted: bool = False):
        """
        Get the number of messages in this session.

        Args:
            include_deleted: Whether to include soft-deleted messages

        Returns:
            Number of messages
        """
        if include_deleted:
            return self.messages.count()
        else:
            return self.messages.filter(is_deleted=False).count()


class ChatHistory(models.Model):
    """
    Stores individual chat messages within a session.

    Attributes:
        session (ChatSession): Related chat session.
        content (str): Message content.
        role (str): Message role (user, assistant, system).
        timestamp (datetime): Message creation time.
        token_count (int): Optional token count.
        order (int): Order for sorting messages.
        is_deleted (bool): Whether the message has been soft deleted.
    """

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    content = models.TextField()
    role = models.CharField(
        choices=RoleChoices.choices,
        default=RoleChoices.USER,
        max_length=10,
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    token_count = models.IntegerField(
        null=True,
        blank=True,
    )
    order = models.IntegerField(
        _("order"),
        default=0,
        help_text=_("For ordering in case of simultaneous writes"),
    )
    is_deleted = models.BooleanField(
        default=False,
        help_text="Whether the message has been soft deleted for privacy compliance",
        db_index=True,
    )

    objects = ChatHistoryManager()

    class Meta:
        verbose_name = _("Chat Message")
        verbose_name_plural = _("Chat Messages")
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["session", "timestamp"]),
            models.Index(fields=["session", "is_deleted"]),  # For filtering non-deleted messages
            models.Index(fields=["is_deleted", "timestamp"]),  # For cleanup operations
        ]

    def __str__(self) -> str:
        """
        Return a string representation of the chat message.
        """
        return f"{self.role}: {self.content[:50]}..."

    def soft_delete(self) -> None:
        """Soft delete this message by setting is_deleted to True."""
        self.is_deleted = True
        self.save(update_fields=["is_deleted"])

    def restore(self) -> None:
        """Restore this message by setting is_deleted to False."""
        self.is_deleted = False
        self.save(update_fields=["is_deleted"])

    def to_langchain_message(self):
        """
        Convert this ChatHistory instance to a LangChain message.

        Returns:
            Appropriate LangChain message object (HumanMessage, AIMessage, SystemMessage)
        """

        if self.role == "USER":
            return HumanMessage(content=self.content)
        elif self.role == "ASSISTANT":
            return AIMessage(content=self.content)
        elif self.role == "SYSTEM":
            return SystemMessage(content=self.content)
        else:
            # Default to HumanMessage for unknown roles
            return HumanMessage(content=self.content)

    @classmethod
    def from_langchain_message(cls, session, message, order: Optional[int] = None, **kwargs):
        """
        Create a ChatHistory instance from a LangChain message.

        Args:
            session: The ChatSession instance
            message: LangChain message object
            order: Message order (auto-calculated if not provided)
            **kwargs: Additional fields for the ChatHistory model

        Returns:
            Created ChatHistory instance
        """
        # Map LangChain message types to Django model roles
        role_mapping = {
            "human": "USER",
            "ai": "ASSISTANT",
            "system": "SYSTEM",
            "function": "SYSTEM",  # Map function messages to system
        }

        role = role_mapping.get(message.type, "USER")

        # Auto-calculate order if not provided
        if order is None:
            last_message = session.messages.order_by("-order").first()
            order = (last_message.order + 1) if last_message else 0

        return cls.objects.create(
            session=session, content=message.content, role=role, order=order, **kwargs
        )

    def update_content(self, new_content: str):
        """
        Update the content of this message.

        Args:
            new_content: New message content
        """
        self.content = new_content
        self.save(
            update_fields=["content", "updated_at"] if hasattr(self, "updated_at") else ["content"]
        )

    def get_context_window(self, window_size: int = 5):
        """
        Get a context window of messages around this message.

        Args:
            window_size: Number of messages before and after this message

        Returns:
            QuerySet of ChatHistory messages in the context window
        """
        return ChatHistory.objects.filter(
            session=self.session,
            is_deleted=False,
            order__gte=max(0, self.order - window_size),
            order__lte=self.order + window_size,
        ).order_by("order")


class InteractionManager(models.Manager):
    """
    Manager for Interaction model, providing helper methods for creation and filtering.
    """

    def completed_interactions(self):
        """
        Return queryset of completed (successful) interactions.
        """
        return self.filter(status="success")

    def for_session(self, session_id):
        """
        Return queryset of interactions for a given session ID.
        """
        return self.filter(session_id=session_id)


class AbstractInteractionLog(TimeStampedModel):
    """
    Logs LLM interactions for auditing, cost analysis, and debugging.

    Attributes:
        user (User): User who initiated the interaction.
        workflow (Workflow): The associated workflow
        prompt_text (str): Prompt sent to the LLM.
        response_text (str): LLM response.
        model_name (str): Name of the LLM model used.
        provider (str): LLM provider.
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
        total_cost (Decimal): Estimated cost in USD.
        latency_ms (int): Latency in milliseconds.
        status (str): Success or error.
        error_message (str): Error message if failed.
        created_at (datetime): Creation timestamp.
        metadata (dict): Additional metadata.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    workflow = models.ForeignKey(AbstractWorkflow, on_delete=models.SET_NULL, null=True, blank=True)
    prompt_text = models.JSONField(default=dict, null=True, blank=True)
    response_text = models.TextField(null=True, blank=True)
    model_name = models.CharField(
        max_length=100, help_text="Name of the LLM model used", null=False, blank=False
    )
    provider = models.CharField(max_length=50, null=False, blank=False)
    input_tokens = models.IntegerField(null=True, blank=True)
    output_tokens = models.IntegerField(null=True, blank=True)
    model_parameters = models.JSONField(default=dict, null=True, blank=True)
    latency_ms = models.IntegerField(
        null=True,
        blank=True,
    )
    status = models.CharField(
        max_length=20,
        choices=AIRequestStatus.choices,
        default=AIRequestStatus.PROCESSING,
    )
    error_message = models.TextField(
        blank=True,
        null=True,
    )
    metadata = models.JSONField(
        default=dict,
        null=True,
        blank=True,
    )

    objects = InteractionManager()

    class Meta:
        abstract = True
        verbose_name = _("LLM Interaction Log")
        verbose_name_plural = _("LLM Interaction Logs")
        ordering: tuple[str, ...] = ("-updated_at", "-created_at")

    def __str__(self) -> str:
        """
        Return a string representation of the LLM interaction log.
        """
        return f"Log {self.pk} - {self.model_name} ({self.status})"

    def get_logging_handler(self, handler):
        handlers = []
        if handler == "basic":
            handlers.append(LoggingHandler(interaction_log=self))
        return handlers


class InteractionLog(AbstractInteractionLog):
    workflow = models.ForeignKey(Workflow, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        verbose_name = _("LLM Interaction Log")
        verbose_name_plural = _("LLM Interaction Logs")
        ordering = ["-created_at"]

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the InteractionLog instance for API responses.
        Sanitizes sensitive data to prevent API key leakage.

        Returns:
            dict: Dictionary with sanitized interaction log fields.
        """
        data = model_to_dict(self, exclude=["id"])

        if data.get("model_parameters"):
            data["model_parameters"] = _sanitize_model_parameters(data["model_parameters"])

        if data.get("prompt_text"):
            data["prompt_text"] = _sanitize_sensitive_data(data["prompt_text"])

        if data.get("metadata"):
            data["metadata"] = _sanitize_sensitive_data(data["metadata"])

        formatted_data = {"id": str(self.id), **data}
        return formatted_data

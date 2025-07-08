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
from enum import unique
from functools import reduce
from typing import Any
from typing import Optional

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.db import transaction
from django.db.models import ForeignKey
from django.db.models import Max
from django.forms.models import model_to_dict
from django.utils.translation import gettext_lazy as _
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from django_chain.exceptions import PromptValidationError
from django_chain.exceptions import WorkflowValidationError
from django_chain.providers import get_chat_model
from django_chain.utils.llm_client import LoggingHandler
from django_chain.utils.llm_client import add_wrapper_function
from django_chain.utils.prompts import _convert_to_prompt_template
from django_chain.utils.workflows import _convert_to_runnable
from django_chain.validators import validate_prompt
from django_chain.validators import validate_workflow


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

        # TODO: add validation for input variables against prompt content
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
        # Creating this to format ouput to start with id
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

    def to_langchain_chain(self, *args, **kwargs) -> Any:  # noqa C901
        """
        Constructs and returns a LangChain RunnableSequence from the workflow definition.

        Returns:
            Any: LangChain RunnableSequence instance.

        Raises:
            ImportError: If LangChain is not installed.
            ValueError: If the workflow definition is invalid.
        """
        chain_components = []
        config_override = {"prompt_instance": self.prompt}
        current_config = {
            **kwargs,
            **config_override,
        }
        chain_components = _convert_to_runnable(workflow=self, **current_config)

        logging_toggle = kwargs.get("log")
        if logging_toggle == "true":
            interaction_log = InteractionLog.objects.create(
                workflow=self,
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
        # Creating this to format ouput to start with id
        formatted_data = {"id": str(self.id), **data}
        return formatted_data

    class Meta:
        verbose_name = _("Workflow")
        verbose_name_plural = _("Workflows")
        ordering: tuple[str, ...] = ("-updated_at", "-created_at", "name", "-version")


class AbstractChatSession(TimeStampedModel):
    """
    Stores chat session information, including user, session ID, and LLM config.

    Attributes:
        user (User): Associated user (nullable).
        session_id (str): Unique session identifier.
        title (str): Optional title for the chat session.
        llm_config (dict): LLM configuration for this session.
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
    session_id = models.CharField(
        max_length=100,
        unique=True,
        help_text="UUID for anonymous sessions or custom session tracking",
    )
    title = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        help_text="A user-friendly title for the chat",
    )
    workflow = ForeignKey(AbstractWorkflow, on_delete=models.PROTECT, null=True, blank=True)

    class Meta:
        abstract = True
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"
        ordering: tuple[str, ...] = ("-updated_at", "-created_at")
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
        ]

    def __str__(self) -> str:
        return self.title or f"Chat Session {self.session_id}"


class ChatSession(AbstractChatSession):
    workflow = ForeignKey(Workflow, on_delete=models.PROTECT, null=True, blank=True)

    class Meta:
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"
        ordering: tuple[str, ...] = ("-updated_at", "-created_at")
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
        ]


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

    class Meta:
        verbose_name = _("Chat Message")
        verbose_name_plural = _("Chat Messages")
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["session", "timestamp"]),
        ]

    def __str__(self) -> str:
        """
        Return a string representation of the chat message.
        """
        return f"{self.role}: {self.content[:50]}..."


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
        Returns a dictionary representation of the Workflow instance for API responses.

        Returns:
            dict: Dictionary with workflow fields.
        """
        data = model_to_dict(self, exclude=["id"])
        # Creating this to format ouput to start with id
        formatted_data = {"id": str(self.id), **data}
        return formatted_data

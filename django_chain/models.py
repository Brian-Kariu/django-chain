"""
Chain models for django-chain.
"""

import logging
import uuid
from typing import Any
from typing import Optional

from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import gettext_lazy as _
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from django_chain.exceptions import DjangoChainError
from django_chain.exceptions import PromptValidationError
from django_chain.utils.chain_client import chat_workflow

try:
    from langchain_core.messages import AIMessage
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.prompts import AIMessagePromptTemplate
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts import HumanMessagePromptTemplate
    from langchain_core.prompts import PromptTemplate
    from langchain_core.prompts import SystemMessagePromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain is not installed. Prompt conversion functionality will be disabled.")

User = get_user_model()
LOGGER = logging.getLogger(__name__)


class PromptTemplateTypes(models.TextChoices):
    """
    Available choices of prompt types. Based on langchain types from docs.
    https://python.langchain.com/docs/concepts/prompt_templates/
    """

    PROMPT_TEMPLATE = "PROMPT_TEMPLATE", _("String Prompt Template")
    CHAT_PROMPT_TEMPLATE = "CHAT_PROMPT_TEMPLATE", _("Chat Prompt Template")


class PromptTemplateRendering(models.TextChoices):
    FROM_STRINGS = "FROM_STRINGS", _("String Rendering")
    FROM_TEMPLATE = "FROM_TEMPLATE", _("Template Rendering")


class MessageTypes(models.TextChoices):
    STRING = "STRING", _("String Message")
    MESSAGEPLACEHOLDER = "MESSAGEPLACEHOLDER", _("Placeholder message")


class RoleTypes(models.TextChoices):
    BASE = "FROM_STRINGS", _("Base Message")
    HUMAN = "HUMAN", _("Human Message")
    AI = "AI", _("AI Message")
    SYSTEM = "SYSTEM", _("System Message")
    CHAT = "CHAT", _("Chat Message")


class ParserTypes(models.TextChoices):
    JSON = "JSON", _("Json parser")
    PYDANTIC = "PYDANTIC", _("Pydantic parser")
    STRING = "STRING", _("String parser")


class ChainTypes(models.TextChoices):
    CHAT = "CHAT", _("Chat chain")


class Prompt(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    type = models.CharField(
        max_length=20,
        choices=PromptTemplateTypes.choices,
        default=PromptTemplateTypes.CHAT_PROMPT_TEMPLATE,
        null=False,
        blank=False,
    )
    name = models.CharField(max_length=255, unique=True, null=False)
    # TODO: Consider moving template to messages
    prompt_template = models.JSONField(
        default=dict(),
        help_text=_(
            "JSON representation of the LangChain prompt. Must include 'langchain_type' (e.g., 'PromptTemplate', 'ChatPromptTemplate')."
        ),
    )
    version = models.PositiveIntegerField(default=1, null=False, blank=False)
    is_active = models.BooleanField(default=False)
    input_variables = models.JSONField(
        help_text="Input variables to the prompt", blank=True, null=True
    )
    optional_variables = models.JSONField(
        help_text="Input variables to the prompt", blank=True, null=True
    )

    class Meta:
        verbose_name = _("Prompt")
        verbose_name_plural = _("Prompts")
        unique_together = (("name", "version"), ("name", "is_active"))
        ordering = ["name", "-version"]

    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({'Active' if self.is_active else 'Inactive'})"

    def clean(self):
        super().clean()
        if self.is_active:
            active_prompts = Prompt.objects.filter(name=self.name, is_active=True)
            if self.pk:
                active_prompts = active_prompts.exclude(pk=self.pk)
            if active_prompts.exists():
                raise ValidationError(
                    _(
                        "There can only be one active prompt per name. "
                        "Please deactivate the existing active prompt before setting this one as active."
                    ),
                    code="duplicate_active_prompt",
                )

        # Validate prompt_template structure for conversion
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

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    @classmethod
    def create_new_version(
        cls, name, prompt_template, description="", input_variables=None, activate=True
    ):
        max_version = cls.objects.filter(name=name).aggregate(max("version"))["version__max"]
        new_version_number = (max_version or 0) + 1

        new_prompt = cls(
            name=name,
            version=new_version_number,
            prompt_template=prompt_template,
            description=description,
            input_variables=input_variables,
            is_active=activate,
        )
        new_prompt.full_clean()

        if activate:
            cls.objects.filter(name=name, is_active=True).update(is_active=False)

        new_prompt.save()
        return new_prompt

    def activate(self):
        Prompt.objects.filter(name=self.name, is_active=True).exclude(pk=self.pk).update(
            is_active=False
        )
        self.is_active = True
        self.save()

    def deactivate(self):
        self.is_active = False
        self.save()

    def get_prompt_content(self):
        return self.prompt_template

    def get_input_variables(self):
        return self.input_variables if self.input_variables is not None else []

    def to_langchain_prompt(self):  # noqa: C901
        """
        Converts the stored JSON prompt_template into an actual LangChain prompt object.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain library is not installed. Cannot convert to LangChain prompt object."
            )

        prompt_data = self.prompt_template
        langchain_type = prompt_data.get("langchain_type")

        if not langchain_type:
            raise ValueError("Invalid prompt_template: 'langchain_type' key is missing.")

        if langchain_type == "PromptTemplate":
            template = prompt_data.get("template")
            input_variables = prompt_data.get("input_variables", [])
            if not isinstance(input_variables, list):
                raise ValueError("input_variables for PromptTemplate must be a list.")

            if template is None:
                raise ValueError("PromptTemplate requires a 'template' key.")
            return PromptTemplate(template=template, input_variables=input_variables)

        elif langchain_type == "ChatPromptTemplate":
            messages_data = prompt_data.get("messages")
            global_input_variables = prompt_data.get("input_variables", [])
            if not isinstance(global_input_variables, list):
                raise ValueError("input_variables for ChatPromptTemplate must be a list.")

            if not isinstance(messages_data, list):
                raise ValueError(
                    "ChatPromptTemplate requires a 'messages' key, which must be a list."
                )

            langchain_messages = []
            for msg_data in messages_data:
                message_type = msg_data.get("message_type")
                template = msg_data.get("template")
                msg_input_variables = msg_data.get(
                    "input_variables", []
                )  # Message-specific input_variables

                if template is None:
                    raise ValueError(
                        f"Chat message of type '{message_type}' requires a 'template' key."
                    )
                if not isinstance(msg_input_variables, list):
                    raise ValueError(
                        f"input_variables for chat message of type '{message_type}' must be a list."
                    )

                if message_type == "system":
                    langchain_messages.append(SystemMessagePromptTemplate.from_template(template))
                elif message_type == "human":
                    langchain_messages.append(HumanMessagePromptTemplate.from_template(template))
                elif message_type == "ai":
                    langchain_messages.append(AIMessagePromptTemplate.from_template(template))
                else:
                    raise ValueError(f"Unsupported chat message type: {message_type}")

            return ChatPromptTemplate.from_messages(
                langchain_messages, input_variables=global_input_variables
            )

        else:
            raise ValueError(f"Unsupported LangChain prompt type: {langchain_type}")


class Message(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    type = models.CharField(max_length=20, choices=MessageTypes.choices, null=False, blank=False)
    role = models.CharField(
        max_length=20, choices=RoleTypes.choices, default=RoleTypes.BASE, null=False, blank=False
    )
    content = models.CharField(max_length=255, null=False, blank=False)
    prompt = models.ForeignKey(
        Prompt, on_delete=models.CASCADE, related_name="messages", null=False, blank=False
    )
    # TODO: Add functionality to validate the roles a type has.

    def __str__(self):
        return str(self.content)


class OutputParsers(models.Model):
    # NOTE: Here the user should override this model to join to their custom models
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    prompt = models.OneToOneField(
        Prompt, on_delete=models.CASCADE, related_name="parsers", null=False, blank=False
    )
    type = models.CharField(
        max_length=20,
        choices=ParserTypes.choices,
        null=False,
        blank=False,
        default=ParserTypes.JSON,
    )

    def __str__(self):
        return f"{str(self.prompt.name)}_{str(self.type)}"

    def get_parser(self, **kwargs):
        parser_mapping = {
            ParserTypes.JSON: JsonOutputParser,
            ParserTypes.STRING: StrOutputParser,
            ParserTypes.PYDANTIC: PydanticOutputParser,
        }
        parser_class = parser_mapping.get(self.type)
        if parser_class:
            return parser_class(**kwargs)
        else:
            raise DjangoChainError(f"Unknown parser type {self.type}")


class Workflow(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    name = models.CharField(max_length=255, unique=True, blank=False, null=False)
    prompt = models.OneToOneField(Prompt, on_delete=models.PROTECT, blank=False, null=True)
    chain_type = models.CharField(
        max_length=20, choices=ChainTypes.choices, null=False, blank=False
    )

    def __str__(self):
        return str(self.name)

    def get_chain(self):
        chain_mapping = {"CHAT": chat_workflow}
        return chain_mapping.get(self.chain_type)


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


class RoleChoices(models.TextChoices):
    USER = "USER", _("User template")
    ASSISTANT = "ASSISTANT", _("Assistant Template")
    SYSTEM = "SYSTEM", _("System Template")


class ChatMessage(models.Model):
    """
    A model for storing chat messages.
    """

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    content = models.TextField(_("content"))
    role = models.CharField(
        choices=RoleChoices.choices,
        default=RoleChoices.USER,
        max_length=10,
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

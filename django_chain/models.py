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

User = get_user_model()
LOGGER = logging.getLogger(__name__)


class PromptTemplateTypes(models.TextChoices):
    """
    Available choices of prompt types. Based on langchain types from docs.
    https://python.langchain.com/docs/concepts/prompt_templates/
    """

    PROMPT_TEMPLATE = "PROMPT_TEMPLATE", _("String Prompt Template")
    CHAT_PROMPT_TEMPLATE = "CHAT_PROMPT_TEMPLATE", _("Chat Prompt Template")
    MESSAGE_PLACEHOLDER = "MESSAGE_PLACEHOLDER", _("Message Placeholder")


class PromptTemplateRendering(models.TextChoices):
    FROM_STRINGS = "FROM_STRINGS", _("String Rendering")
    FROM_TEMPLATE = "FROM_TEMPLATE", _("Template Rendering")


class MessageTypes(models.TextChoices):
    BASE = "FROM_STRINGS", _("Base Message")
    HUMAN = "HUMAN", _("Human Message")
    AI = "AI", _("AI Message")
    SYSTEM = "SYSTEM", _("System Message")
    CHAT = "CHAT", _("Chat Message")


class ParserTypes(models.TextChoices):
    JSON = "JSON", _("Json parser")
    PYDANTIC = "PYDANTIC", _("Pydantic parser")
    STRING = "STRING", _("String parser")


class PromptTemplate(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    template = models.CharField(
        max_length=20,
        choices=PromptTemplateTypes,
        null=False,
        blank=False,
        default=PromptTemplateTypes.PROMPT_TEMPLATE,
    )
    rendering = models.CharField(
        max_length=20,
        choices=PromptTemplateRendering,
        null=True,
        blank=True,
        default=PromptTemplateRendering.FROM_STRINGS,
    )

    def __str__(self) -> str:
        return str(self.template)


class Prompt(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    name = models.CharField(max_length=255, unique=True, null=False)
    # TODO: Consider moving template to messages
    template = models.ForeignKey(PromptTemplate, on_delete=models.PROTECT, null=False, blank=False)
    input_variables = models.JSONField(
        help_text="Input variables to the prompt", blank=True, null=True
    )
    optional_variables = models.JSONField(
        help_text="Input variables to the prompt", blank=True, null=True
    )

    def __str__(self) -> str:
        return str(self.name)

    def clean(self, *args, **kwargs):
        messages = self.messages.all()

        # Check if prompts have atleast one message
        if messages is None or messages == []:
            raise PromptValidationError
        super().clean(*args, **kwargs)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)


class Message(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    guid = models.UUIDField(unique=True, null=True, blank=True)
    type = models.CharField(max_length=20, choices=MessageTypes, null=False, blank=False)
    content = models.CharField(max_length=255, null=False, blank=False)
    prompt = models.ForeignKey(
        Prompt, on_delete=models.CASCADE, related_name="messages", null=False, blank=False
    )

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
        max_length=20, choices=ParserTypes, null=False, blank=False, default=ParserTypes.JSON
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


class LLMChain(models.Model):
    """
    A model representing a LangChain chain configuration.
    """

    name = models.CharField(
        max_length=255,
        unique=True,
        help_text=_("A unique name for this chain"),
    )
    prompt_template = models.TextField(
        help_text=_("The template for the prompt, using {variable} syntax"),
    )
    model_name = models.CharField(
        max_length=255,
        help_text=_("The name of the LLM model to use"),
    )
    provider = models.CharField(
        max_length=50,
        default="openai",
        help_text=_("The LLM provider to use (e.g., 'openai', 'google')"),
    )
    temperature = models.FloatField(
        default=0.7,
        help_text=_("The temperature parameter for the model"),
    )
    max_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text=_("Maximum number of tokens to generate"),
    )
    input_variables = models.JSONField(
        default=list,
        help_text=_("List of expected input variable names"),
    )
    output_parser = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text=_("Name of LangChain output parser (e.g., 'JSONOutputParser')"),
    )

    class Meta:
        verbose_name = _("LLM Chain")
        verbose_name_plural = _("LLM Chains")
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    def get_chain(self, llm_client: Optional[Any] = None) -> Any:
        """
        Get the LangChain chain instance.

        Args:
            llm_client: Optional LLM client instance to use

        Returns:
            A configured LangChain chain
        """
        # try:
        #     if llm_client is None:
        #         llm = LLMClient.get_chat_model(
        #             provider=self.provider,
        #             model_name=self.model_name,
        #             temperature=self.temperature,
        #             max_tokens=self.max_tokens,
        #         )
        #     else:
        #         llm = llm_client
        #
        #     prompt = PromptManager.get_langchain_prompt(
        #         self.prompt_template,
        #         input_variables=self.input_variables,
        #     )
        #     return prompt | llm
        # except Exception as e:
        #     raise PromptValidationError(f"Failed to create chain: {e!s}") from e

    def format_prompt(self, context: dict[str, Any]) -> str:
        """
        Format the prompt template with the given context.
        """
        try:
            return self.prompt_template.format(**context)
        except KeyError as e:
            raise PromptValidationError(f"Missing required input variable: {e!s}") from e

    def get_chain_config(self) -> dict[str, Any]:
        """Returns the chain configuration as a dictionary."""
        return {
            "name": self.name,
            "prompt_template": self.prompt_template,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "input_variables": self.input_variables,
            "output_parser": self.output_parser,
        }


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

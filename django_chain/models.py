"""
Chain models for django-chain.
"""

import logging
import uuid
from typing import Any
from typing import Optional

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Max
from django.forms.models import model_to_dict
from django.utils.translation import gettext_lazy as _
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from django_chain.exceptions import DjangoChainError
from django_chain.providers.fake import get_fake_chat_model

try:
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


class Prompt(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    name = models.CharField(max_length=255, unique=True, null=False)
    prompt_template = models.JSONField(
        default=dict,
        null=False,
        blank=False,
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

    def to_dict(self):
        """
        Returns a dictionary representation of the Prompt instance for API responses.
        """
        data = model_to_dict(self, exclude=["id"])
        data["id"] = str(self.id)
        return data

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
    def create_new_version(cls, name, prompt_template, input_variables=None, activate=True):
        max_version = cls.objects.filter(name=name).aggregate(Max("version"))["version__max"]
        new_version_number = (max_version or 0) + 1

        new_prompt = cls(
            name=name,
            version=new_version_number,
            prompt_template=prompt_template,
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
                msg_input_variables = msg_data.get("input_variables", [])

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

            return ChatPromptTemplate.from_messages(langchain_messages)

        else:
            raise ValueError(f"Unsupported LangChain prompt type: {langchain_type}")


class Workflow(models.Model):
    """
    Represents an AI workflow, defined as a sequence of LangChain components.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(
        max_length=255,
        unique=True,
        help_text=_(
            "A unique name for this workflow (e.g., 'SummaryGenerator', 'CustomerServiceChatbot')."
        ),
    )
    description = models.TextField(
        blank=True, help_text=_("A brief description of what this workflow does.")
    )
    workflow_definition = models.JSONField(
        default=dict,
        null=False,
        blank=False,
        help_text=_(
            "JSON array defining the sequence of LangChain components (prompt, llm, parser)."
        ),
    )
    is_active = models.BooleanField(
        default=False,
        help_text=_("Only one workflow with a given 'name' should be active at any time."),
    )

    class Meta:
        verbose_name = _("Workflow")
        verbose_name_plural = _("Workflows")
        unique_together = (("name", "is_active"),)
        ordering = ["name"]

    def to_dict(self):
        """
        Returns a dictionary representation of the Workflow instance for API responses.
        """
        data = model_to_dict(self, exclude=["id"])
        data["id"] = str(self.id)
        return data

    def __str__(self):
        return f"{self.name} ({'Active' if self.is_active else 'Inactive'})"

    def clean(self):
        """
        Custom validation for workflow definition and active status.
        """
        super().clean()

        # NOTE: Need to save this activate/deactivate operations in a manager
        if self.is_active:
            active_workflows = Workflow.objects.filter(name=self.name, is_active=True)
            if self.pk:
                active_workflows = active_workflows.exclude(pk=self.pk)
            if active_workflows.exists():
                raise ValidationError(
                    _("There can only be one active workflow with the same name."),
                    code="duplicate_active_workflow",
                )

        if not isinstance(self.workflow_definition, list):
            raise ValidationError(
                _("Workflow definition must be a JSON array (list of steps)."),
                code="invalid_workflow_definition_format",
            )

        for i, step in enumerate(self.workflow_definition):
            if not isinstance(step, dict):
                raise ValidationError(
                    _("Each step in the workflow definition must be a JSON object."),
                    code=f"invalid_step_format_{i}",
                )
            if "type" not in step:
                raise ValidationError(
                    _("Each workflow step must have a 'type' (e.g., 'prompt', 'llm', 'parser')."),
                    code=f"missing_step_type_{i}",
                )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def activate(self):
        """Activates this workflow, deactivating any other active workflow with the same name."""
        Workflow.objects.filter(name=self.name, is_active=True).exclude(pk=self.pk).update(
            is_active=False
        )
        self.is_active = True
        self.save()

    def deactivate(self):
        """Deactivates this workflow."""
        self.is_active = False
        self.save()

    def to_langchain_chain(self):  # noqa: C901
        """
        Constructs and returns a LangChain RunnableSequence from the workflow definition.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain library is not installed. Cannot construct chain.")

        chain_components = []
        for i, step_data in enumerate(self.workflow_definition):
            step_type = step_data.get("type")

            if step_type == "prompt":
                prompt_name = step_data.get("name")
                if not prompt_name:
                    raise ValueError(f"Workflow step {i}: Prompt step requires 'name'.")
                try:
                    prompt_record = Prompt.objects.get(name__iexact=prompt_name, is_active=True)
                    chain_components.append(prompt_record.to_langchain_prompt())
                except Prompt.DoesNotExist:
                    raise ValueError(f"Workflow step {i}: Active prompt '{prompt_name}' not found.")
                except Exception as e:
                    raise ValueError(
                        f"Workflow step {i}: Error loading prompt '{prompt_name}': {e}"
                    )

            elif step_type == "llm":
                llm_config_override = step_data.get("config", {})
                global_llm_config = settings.DJANGO_LLM_SETTINGS

                current_llm_config = {**global_llm_config, **llm_config_override}

                llm_provider = current_llm_config.get("DEFAULT_LLM_PROVIDER")
                model_name = current_llm_config["DEFAULT_CHAT_MODEL"]["name"]
                temperature = current_llm_config["DEFAULT_CHAT_MODEL"]["temperature"]
                api_key = current_llm_config["DEFAULT_CHAT_MODEL"][
                    f"{llm_provider.upper()}_API_KEY"
                ]
                if not llm_provider or not model_name:
                    raise ValueError(
                        f"Workflow step {i}: LLM step requires 'provider' and 'model_name'."
                    )

                llm_instance = None
                if llm_provider == "fake":
                    if not api_key:
                        raise ValueError(
                            f"Workflow step {i}:  LLM requires 'api_key' in LLM config."
                        )
                    llm_instance = get_fake_chat_model()

                else:
                    raise ValueError(f"Workflow step {i}: Unsupported LLM provider: {llm_provider}")

                if llm_instance:
                    chain_components.append(llm_instance)

            elif step_type == "parser":
                parser_type = step_data.get("parser_type")
                parser_args = step_data.get("parser_args", {})

                if not parser_type:
                    raise ValueError(f"Workflow step {i}: Parser step requires 'parser_type'.")

                parser_instance = None
                if parser_type == "StrOutputParser":
                    parser_instance = StrOutputParser(**parser_args)
                elif parser_type == "JsonOutputParser":
                    parser_instance = JsonOutputParser(**parser_args)
                else:
                    raise ValueError(f"Workflow step {i}: Unsupported parser type: {parser_type}")

                if parser_instance:
                    chain_components.append(parser_instance)

            else:
                raise ValueError(f"Workflow step {i}: Unknown component type: {step_type}")

        if not chain_components:
            raise ValueError("Workflow definition is empty or contains no valid components.")

        from functools import reduce

        return reduce(lambda a, b: a | b, chain_components)


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


class UserInteractionManager(models.Manager):
    def create_for_workflow(self, workflow, input_data, user_identifier, session_id):
        return self.create(
            workflow=workflow,
            input_data=input_data,
            user_identifier=user_identifier,
            session_id=session_id,
            status="processing",
        )

    def completed_interactions(self):
        return self.filter(status="success")

    def for_session(self, session_id):
        return self.filter(session_id=session_id)


class UserInteraction(models.Model):
    """
    Records a single overall user query and the final LLM response.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(
        "Workflow",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("The workflow that was executed for this interaction."),
    )
    user_identifier = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text=_("Identifier for the user (e.g., session ID, user ID if authenticated)."),
    )
    session_id = models.UUIDField(
        default=uuid.uuid4,
        blank=True,
        null=True,
        help_text=_("A unique ID to group related user interactions across a session."),
    )
    input_data = models.JSONField(
        help_text=_("The full input JSON received from the user for this interaction.")
    )
    llm_output = models.JSONField(
        blank=True, null=True, help_text=_("The final output JSON/text from the LLM workflow.")
    )
    total_cost_estimate = models.DecimalField(
        max_digits=10,
        decimal_places=8,
        default=0.0,
        help_text=_("Estimated total cost for this interaction (e.g., based on token usage)."),
    )
    total_duration_ms = models.PositiveIntegerField(
        default=0, help_text=_("Total execution time for the workflow in milliseconds.")
    )
    status = models.CharField(
        max_length=50,
        default="success",
        help_text=_("Overall status of the interaction (e.g., 'success', 'failure')."),
    )
    error_message = models.TextField(
        blank=True, null=True, help_text=_("Detailed error message if the interaction failed.")
    )

    objects = UserInteractionManager()

    class Meta:
        verbose_name = _("User Interaction")
        verbose_name_plural = _("User Interactions")
        ordering = ["-id"]

    def __str__(self):
        return f"Interaction {self.id} for {self.workflow.name if self.workflow else 'N/A'}"

    def update_status_and_metrics(
        self,
        status,
        llm_output=None,
        total_cost_estimate=None,
        total_duration_ms=None,
        error_message=None,
    ):
        self.status = status
        if llm_output is not None:
            self.llm_output = llm_output
        if total_cost_estimate is not None:
            self.total_cost_estimate = total_cost_estimate
        if total_duration_ms is not None:
            self.total_duration_ms = total_duration_ms
        if error_message is not None:
            self.error_message = error_message
        self.save()


class InteractionLogManager(models.Manager):
    def create_step_log(
        self,
        user_interaction,
        step_order,
        step_type,
        component_name,
        input_to_step,
        output_from_step,
        metadata,
        duration_ms,
        status="success",
        error_message=None,
    ):
        return self.create(
            user_interaction=user_interaction,
            step_order=step_order,
            step_type=step_type,
            component_name=component_name,
            input_to_step=input_to_step,
            output_from_step=output_from_step,
            metadata=metadata,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
        )


class InteractionLog(models.Model):
    """
    Records detailed step-by-step information for each component within a workflow execution.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_interaction = models.ForeignKey(
        "UserInteraction",
        on_delete=models.CASCADE,
        related_name="logs",
        help_text=_("The parent user interaction this log entry belongs to."),
    )
    step_order = models.PositiveIntegerField(
        help_text=_("The sequential order of this step within the workflow execution.")
    )
    step_type = models.CharField(
        max_length=50,
        help_text=_("Type of the LangChain component (e.g., 'prompt', 'llm', 'parser')."),
    )
    component_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text=_("Name of the specific component used (e.g., Prompt name, LLM model name)."),
    )
    input_to_step = models.JSONField(
        blank=True,
        null=True,
        help_text=_("The input payload received by this specific component in the chain."),
    )
    output_from_step = models.JSONField(
        blank=True,
        null=True,
        help_text=_("The output payload generated by this specific component in the chain."),
    )
    metadata = models.JSONField(
        blank=True,
        null=True,
        help_text=_(
            "Additional step-specific metadata (e.g., token usage for LLM, finish reason)."
        ),
    )
    duration_ms = models.PositiveIntegerField(
        default=0, help_text=_("Time taken for this specific step in milliseconds.")
    )
    status = models.CharField(
        max_length=50,
        default="success",
        help_text=_("Status of this individual step ('success', 'failure')."),
    )
    error_message = models.TextField(
        blank=True, null=True, help_text=_("Error message if this step failed.")
    )

    objects = InteractionLogManager()

    class Meta:
        verbose_name = _("Interaction Log")
        verbose_name_plural = _("Interaction Logs")
        ordering = ["user_interaction", "step_order"]
        unique_together = (("user_interaction", "step_order"),)

    def __str__(self):
        return f"Log for {self.user_interaction} - Step {self.step_order} ({self.step_type})"

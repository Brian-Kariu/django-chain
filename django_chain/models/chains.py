"""
Chain models for django-chain.
"""

from typing import Any, Optional

from django.db import models
from django.utils.translation import gettext_lazy as _

from django_chain.exceptions import PromptValidationError
from django_chain.models.common import AbstractBase
from django_chain.services.llm_client import LLMClient
from django_chain.services.prompt_manager import PromptManager


class LLMChain(AbstractBase):
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
        try:
            if llm_client is None:
                llm = LLMClient.get_chat_model(
                    provider=self.provider,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                llm = llm_client

            prompt = PromptManager.get_langchain_prompt(
                self.prompt_template,
                input_variables=self.input_variables,
            )
            return prompt | llm
        except Exception as e:
            raise PromptValidationError(f"Failed to create chain: {e!s}") from e

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

"""
Prompt manager service for handling prompt templates.
"""

import logging
from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate as LC_PromptTemplate

from django_chain.exceptions import PromptValidationError

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Service for managing prompt templates.
    """

    _output_parsers = {
        "str": StrOutputParser(),
        "json": JsonOutputParser(),
    }

    @classmethod
    def get_langchain_prompt(
        cls,
        template: str,
        input_variables: Optional[list[str]] = None,
        output_parser: Optional[str] = None,
        **kwargs: Any,
    ) -> LC_PromptTemplate:
        """
        Create a LangChain PromptTemplate from a template string.

        Args:
            template: The prompt template string
            input_variables: List of expected input variable names
            output_parser: Name of the output parser to use
            **kwargs: Additional arguments to pass to PromptTemplate

        Returns:
            A configured LangChain PromptTemplate

        Raises:
            PromptValidationError: If the template is invalid or missing required variables
        """
        try:
            # If input_variables is not provided, try to extract them from the template
            if input_variables is None:
                # Simple extraction of {variable} patterns
                import re

                input_variables = re.findall(r"{([^}]+)}", template)

            prompt = LC_PromptTemplate(template=template, input_variables=input_variables, **kwargs)

            # Add output parser if specified
            if output_parser:
                if output_parser not in cls._output_parsers:
                    raise PromptValidationError(f"Unsupported output parser: {output_parser}")
                prompt = prompt | cls._output_parsers[output_parser]

            return prompt

        except Exception as e:
            logger.error(f"Error creating prompt template: {e}", exc_info=True)
            raise PromptValidationError(f"Failed to create prompt template: {e!s}") from e

    @classmethod
    def register_output_parser(
        cls,
        name: str,
        parser: Any,
    ) -> None:
        """
        Register a custom output parser.

        Args:
            name: The name to register the parser under
            parser: The output parser instance
        """
        cls._output_parsers[name] = parser

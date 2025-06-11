"""
Custom exceptions for django-chain.
"""


class DjangoChainError(Exception):
    """Base exception for django-chain related errors."""


class LLMProviderAPIError(DjangoChainError):
    """Raised when there's an issue communicating with the LLM provider."""


class LLMResponseError(DjangoChainError):
    """Raised when the LLM returns an unexpected or invalid response."""


class PromptValidationError(DjangoChainError):
    """Raised when a prompt template is used incorrectly."""


class ChainExecutionError(DjangoChainError):
    """Raised for general errors during LangChain chain execution."""


class VectorStoreError(DjangoChainError):
    """Raised for errors during vector store operations."""


class MissingDependencyError(DjangoChainError):
    """
    Raised when a required dependency for a specific integration is not installed.

    Attributes:
        integration (str): The name of the integration that requires the missing dependency.
        package (str): The name of the missing package.
        hint (str): A helpful message suggesting how to install the missing dependency.
    """

    def __init__(self, integration: str, package: str):
        self.integration = integration
        self.package = package
        self.hint = f"Try running: pip install django-chain[{integration}]"
        message = (
            f"Required {integration} integration package '{package}' is not installed.\n{self.hint}"
        )
        super().__init__(message)

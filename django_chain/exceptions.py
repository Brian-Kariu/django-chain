"""
Custom exceptions for django-chain.
"""


class DjangoChainError(Exception):
    """Base exception for django-chain related errors."""

    def __init__(
        self,
        value,
        message: str = f"Error with Django Chain package: ",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message}"
        self.value = value
        self.expected_format = expected_format
        self.additional_message = additional_message
        if hasattr(self, "expected_format"):
            self.message = self.message + f"Expected {expected_format}. "
        if hasattr(self, "additional_message"):
            self.message = self.message + f"{additional_message}"
        super().__init__(self.value, self.message)


class LLMProviderAPIError(DjangoChainError):
    """Raised when there's an issue communicating with the LLM provider."""

    def __init__(
        self,
        value,
        message: str = f"Error with LLM Vendor: ",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message}"
        self.value = value
        super().__init__(self.value, self.message)


class LLMResponseError(DjangoChainError):
    """Raised when the LLM returns an unexpected or invalid response."""

    def __init__(
        self,
        value,
        message: str = f"Error during with LLM response Generation: ",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message}"
        self.value = value
        super().__init__(self.value, self.message)


class PromptValidationError(DjangoChainError, ValueError):
    """Raised when a prompt template is used incorrectly."""

    def __init__(
        self,
        value,
        message: str = "Invalid prompt input:",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message} '{value}'. "
        super().__init__(value, self.message, expected_format, additional_message)


class WorkflowValidationError(DjangoChainError, ValueError):
    """Raised when a workflow definition is used incorrectly."""

    def __init__(
        self,
        value,
        message: str = "Invalid workflow input:",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message} '{value}'. "
        super().__init__(value, self.message, expected_format, additional_message)


class ChainExecutionError(DjangoChainError):
    """Raised for general errors during LangChain chain execution."""

    def __init__(
        self,
        value,
        message: str = f"Error during Langchain Execution: ",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message}"
        self.value = value
        super().__init__(self.value, self.message)


class VectorStoreError(DjangoChainError):
    """Raised for errors during vector store operations."""

    def __init__(
        self,
        value,
        message: str = f"Error with Vector Store: ",
        expected_format: str = "",
        additional_message: str | tuple[str] = "",
    ):
        self.message = f"{message}"
        self.value = value
        super().__init__(self.value, self.message)


class MissingDependencyError(Exception):
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
        self.message = (
            f"Required {integration} integration package '{package}' is not installed.\n{self.hint}"
        )
        super().__init__(self.message)

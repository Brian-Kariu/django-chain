import pytest
from django_chain.exceptions import (
    DjangoChainError,
    LLMProviderAPIError,
    LLMResponseError,
    PromptValidationError,
    ChainExecutionError,
    VectorStoreError,
    MissingDependencyError,
)


class TestExceptions:
    """Test custom exception hierarchy and functionality."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            LLMProviderAPIError,
            LLMResponseError,
            PromptValidationError,
            ChainExecutionError,
            VectorStoreError,
        ],
    )
    def test_exception_inheritance(self, exception_class):
        """Test that all custom exceptions inherit from DjangoChainError."""
        assert issubclass(exception_class, DjangoChainError)

    @pytest.mark.parametrize(
        "exception_class,message",
        [
            (LLMProviderAPIError, "Error with LLM Vendor:"),
            (LLMResponseError, "Error during with LLM response Generation:"),
            (PromptValidationError, "Invalid prompt input:"),
            (ChainExecutionError, "Error during Langchain Execution:"),
            (VectorStoreError, "Error with Vector Store:"),
            (MissingDependencyError, "is not installed"),
        ],
    )
    def test_exception_instantiation(self, exception_class, message):
        """Test that exceptions can be instantiated with messages."""
        if exception_class == MissingDependencyError:
            exc = exception_class(integration="message", package="custom_package")
            assert message in str(exc.message)
        else:
            exc = exception_class(value="invalid", message=message)
            assert message in str(exc.message)
            assert isinstance(exc, DjangoChainError)

    def test_exception_with_additional_data(self):
        """Test exceptions that accept additional data."""
        exc = PromptValidationError(
            value={"invalid": "template"}, additional_message="Missing langchain_type"
        )
        assert "Missing langchain_type" in str(exc)
        assert exc.value == {"invalid": "template"}

    def test_base_exception_functionality(self):
        """Test base DjangoChainError functionality."""
        exc = DjangoChainError("Base error message")
        assert str(exc.value) == "Base error message"
        assert isinstance(exc, Exception)

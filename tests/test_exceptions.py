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


def test_all_exceptions_are_subclasses():
    assert issubclass(LLMProviderAPIError, DjangoChainError)
    assert issubclass(LLMResponseError, DjangoChainError)
    assert issubclass(PromptValidationError, DjangoChainError)
    assert issubclass(ChainExecutionError, DjangoChainError)
    assert issubclass(VectorStoreError, DjangoChainError)
    assert issubclass(MissingDependencyError, DjangoChainError)


@pytest.mark.parametrize(
    "exc_class",
    [
        LLMProviderAPIError,
        LLMResponseError,
        PromptValidationError,
        ChainExecutionError,
        VectorStoreError,
    ],
)
def test_basic_exceptions_raise(exc_class):
    with pytest.raises(exc_class):
        raise exc_class("test message")


def test_missing_dependency_error_attributes_and_message():
    with pytest.raises(MissingDependencyError) as exc:
        raise MissingDependencyError(integration="openai", package="openai")

    err = exc.value
    assert err.integration == "openai"
    assert err.package == "openai"
    assert "Required openai integration package 'openai' is not installed" in str(err)
    assert "pip install django-chain[openai]" in str(err)
    assert err.hint == "Try running: pip install django-chain[openai]"

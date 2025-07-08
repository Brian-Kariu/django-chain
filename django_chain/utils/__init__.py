"""
Utility functions and helpers for django-chain.

This package contains utility modules for LLM client instantiation, serialization, and workflow execution.
"""

import logging

LOGGER = logging.getLogger(__name__)
try:
    from langchain_core.prompts import AIMessagePromptTemplate
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts import HumanMessagePromptTemplate
    from langchain_core.prompts import PromptTemplate
    from langchain_core.prompts import SystemMessagePromptTemplate

    LANGCHAIN_AVAILABLE = True

except ImportError:
    LANGCHAIN_AVAILABLE = False
    LOGGER.warning(
        "Warning: LangChain is not installed. Various utility functionalities will be disabled, please ensure you have installed the latest langchain version"
    )

# TODO: Add an __all__ method here

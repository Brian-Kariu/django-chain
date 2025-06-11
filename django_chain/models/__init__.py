"""
Models for django-chain.
"""

from django.apps import apps


def get_llm_chain_model():
    """Get the LLMChain model."""
    return apps.get_model("django_chain", "LLMChain")


def get_chat_session_model():
    """Get the ChatSession model."""
    return apps.get_model("django_chain", "ChatSession")


def get_chat_message_model():
    """Get the ChatMessage model."""
    return apps.get_model("django_chain", "ChatMessage")


def get_llm_interaction_log_model():
    """Get the LLMInteractionLog model."""
    return apps.get_model("django_chain", "LLMInteractionLog")


# Export model getters
__all__ = [  # Model getters
    "get_llm_chain_model",
    "get_chat_session_model",
    "get_chat_message_model",
    "get_llm_interaction_log_model",
]

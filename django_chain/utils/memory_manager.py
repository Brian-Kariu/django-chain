"""
Memory manager service for handling chat history with Django model integration.

This module provides LangChain-compatible memory implementations that use Django's
ChatHistory model as the primary storage mechanism, eliminating the need for
separate in-memory storage and providing better persistence and thread safety.
"""

import logging
from typing import Any, Optional, Union, Type
from abc import ABC, abstractmethod

import uuid
from django.apps import apps
from django.core.exceptions import ObjectDoesNotExist
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from django_chain.exceptions import ChainExecutionError
from django_chain.config import app_settings

logger = logging.getLogger(__name__)


class DjangoModelChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """
    LangChain-compatible chat message history using Django's ChatHistory model.

    This implementation uses the ChatHistory model as the primary storage,
    eliminating the need for separate in-memory storage and providing
    better persistence, thread safety, and integration with Django.
    """

    session_id: Union[str, uuid.UUID] = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, session_id: Union[str, uuid.UUID], **kwargs):
        super().__init__(session_id=session_id, **kwargs)
        self._session = None

    @property
    def session(self):
        """Lazy-load the ChatSession to avoid database hits during initialization."""
        if self._session is None:
            try:
                ChatSession = apps.get_model("django_chain", "ChatSession")
                session_uuid = (
                    uuid.UUID(str(self.session_id))
                    if isinstance(self.session_id, str)
                    else self.session_id
                )
                self._session = ChatSession.objects.get(session_id=session_uuid)
            except (ObjectDoesNotExist, ValueError) as e:
                logger.error(f"Session {self.session_id} not found: {e}")
                raise ChainExecutionError(f"Chat session {self.session_id} not found") from e
        return self._session

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message to the chat history."""
        self.add_messages([message])

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages to the chat history."""
        ChatHistory = apps.get_model("django_chain", "ChatHistory")

        # Map LangChain message types to Django model roles
        role_mapping = {
            "human": "USER",
            "ai": "ASSISTANT",
            "system": "SYSTEM",
            "function": "SYSTEM",  # Map function messages to system
        }

        # Get the current max order to maintain message sequence
        last_message = self.session.messages.order_by("-order").first()
        current_order = (last_message.order + 1) if last_message else 0

        # Create ChatHistory objects for each message
        chat_messages = []
        for i, message in enumerate(messages):
            role = role_mapping.get(message.type, "USER")
            chat_messages.append(
                ChatHistory(
                    session=self.session,
                    content=message.content,
                    role=role,
                    order=current_order + i,
                )
            )

        # Bulk create for efficiency
        ChatHistory.objects.bulk_create(chat_messages)
        logger.debug(f"Added {len(messages)} messages to session {self.session_id}")

    def get_messages(self) -> list[BaseMessage]:
        """Retrieve all messages from the chat history as LangChain messages."""
        messages = []

        # Get messages ordered by timestamp and order
        chat_messages = self.session.messages.filter(is_deleted=False).order_by(
            "timestamp", "order"
        )

        # Convert Django model instances to LangChain messages
        for msg in chat_messages:
            if msg.role == "USER":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "ASSISTANT":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "SYSTEM":
                messages.append(SystemMessage(content=msg.content))
            else:
                # Default to HumanMessage for unknown roles
                messages.append(HumanMessage(content=msg.content))

        return messages

    def clear(self) -> None:
        """Clear all messages from the chat history (soft delete)."""
        updated_count = self.session.messages.filter(is_deleted=False).update(is_deleted=True)
        logger.info(f"Soft deleted {updated_count} messages from session {self.session_id}")


class BaseMemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @abstractmethod
    def get_chat_history(self, session_id: Union[str, uuid.UUID]) -> BaseChatMessageHistory:
        """Get a chat message history instance for the given session."""
        pass

    @abstractmethod
    def get_memory(
        self, session_id: Union[str, uuid.UUID], memory_type: str = "buffer", **kwargs
    ) -> Union[ConversationBufferMemory, ConversationBufferWindowMemory]:
        """Get a configured LangChain memory object."""
        pass


class DjangoMemoryProvider(BaseMemoryProvider):
    """Memory provider using Django's ChatHistory model."""

    def get_chat_history(self, session_id: Union[str, uuid.UUID]) -> BaseChatMessageHistory:
        """Get a Django model-backed chat message history."""
        return DjangoModelChatMessageHistory(session_id=session_id)

    def get_memory(
        self,
        session_id: Union[str, uuid.UUID],
        memory_type: str = "buffer",
        k: Optional[int] = None,
        **kwargs,
    ) -> Union[ConversationBufferMemory, ConversationBufferWindowMemory]:
        """
        Get a LangChain memory object backed by Django's ChatHistory model.

        Args:
            session_id: The chat session identifier
            memory_type: Type of memory ('buffer' or 'buffer_window')
            k: Number of messages to keep for window memory
            **kwargs: Additional arguments passed to memory constructor

        Returns:
            Configured LangChain memory object
        """
        chat_history = self.get_chat_history(session_id)

        if memory_type == "buffer":
            return ConversationBufferMemory(
                chat_memory=chat_history, return_messages=True, **kwargs
            )
        elif memory_type == "buffer_window":
            return ConversationBufferWindowMemory(
                chat_memory=chat_history, k=k or 5, return_messages=True, **kwargs
            )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")


class LegacyInMemoryProvider(BaseMemoryProvider):
    """
    Legacy in-memory provider for backward compatibility.

    WARNING: This provider is not thread-safe and should only be used
    for testing or simple single-threaded applications.
    """

    def __init__(self):
        self._store = {}
        logger.warning(
            "Using LegacyInMemoryProvider. Consider migrating to DjangoMemoryProvider "
            "for better persistence and thread safety."
        )

    def get_chat_history(self, session_id: Union[str, uuid.UUID]) -> BaseChatMessageHistory:
        """Get an in-memory chat message history."""
        from langchain_community.chat_message_histories import ChatMessageHistory

        session_key = str(session_id)
        if session_key not in self._store:
            self._store[session_key] = ChatMessageHistory()
        return self._store[session_key]

    def get_memory(
        self,
        session_id: Union[str, uuid.UUID],
        memory_type: str = "buffer",
        k: Optional[int] = None,
        **kwargs,
    ) -> Union[ConversationBufferMemory, ConversationBufferWindowMemory]:
        """Get a LangChain memory object with in-memory storage."""
        chat_history = self.get_chat_history(session_id)

        if memory_type == "buffer":
            return ConversationBufferMemory(
                chat_memory=chat_history, return_messages=True, **kwargs
            )
        elif memory_type == "buffer_window":
            return ConversationBufferWindowMemory(
                chat_memory=chat_history, k=k or 5, return_messages=True, **kwargs
            )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")


class MemoryManager:
    """
    Central memory manager that provides access to different memory providers.

    This class acts as a factory for memory providers and provides a unified
    interface for memory operations across the django-chain application.
    """

    _providers = {
        "django": DjangoMemoryProvider,
        "inmemory": LegacyInMemoryProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseMemoryProvider]):
        """Register a custom memory provider."""
        cls._providers[name] = provider_class
        logger.info(f"Registered memory provider: {name}")

    @classmethod
    def get_provider(cls, provider_name: Optional[str] = None) -> BaseMemoryProvider:
        """
        Get a memory provider instance.

        Args:
            provider_name: Name of the provider to use. If None, uses the default
                          from settings or falls back to 'django'.

        Returns:
            Memory provider instance
        """
        if provider_name is None:
            # Get from settings with fallback to 'django'
            memory_settings = getattr(app_settings, "MEMORY", {})
            provider_name = memory_settings.get("PROVIDER", "django")

        if provider_name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown memory provider '{provider_name}'. Available providers: {available}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class()


# Convenience functions for backward compatibility and ease of use
def get_chat_history(
    session_id: Union[str, uuid.UUID], provider: Optional[str] = None
) -> BaseChatMessageHistory:
    """
    Get a chat message history for the given session.

    Args:
        session_id: The chat session identifier
        provider: Memory provider to use (defaults to configured provider)

    Returns:
        Chat message history instance
    """
    memory_provider = MemoryManager.get_provider(provider)
    return memory_provider.get_chat_history(session_id)


def get_langchain_memory(
    session_id: Union[str, uuid.UUID],
    memory_type: str = "buffer",
    k: Optional[int] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> Union[ConversationBufferMemory, ConversationBufferWindowMemory]:
    """
    Get a LangChain memory object for the given session.

    Args:
        session_id: The chat session identifier
        memory_type: Type of memory ('buffer' or 'buffer_window')
        k: Number of messages to keep for window memory
        provider: Memory provider to use (defaults to configured provider)
        **kwargs: Additional arguments passed to memory constructor

    Returns:
        Configured LangChain memory object

    Raises:
        ChainExecutionError: If there's an error creating the memory
    """
    try:
        memory_provider = MemoryManager.get_provider(provider)
        return memory_provider.get_memory(
            session_id=session_id, memory_type=memory_type, k=k, **kwargs
        )
    except Exception as e:
        logger.error(f"Error creating memory: {e}", exc_info=True)
        raise ChainExecutionError(f"Failed to create memory: {e!s}") from e


def save_messages_to_session(
    session_id: Union[str, uuid.UUID],
    messages: list[Union[dict[str, Any], BaseMessage]],
) -> None:
    """
    Save a list of LangChain messages to the chat session.

    Args:
        session_id: The chat session identifier
        messages: The list of LangChain messages or dictionaries to save

    Raises:
        ChainExecutionError: If there's an error saving the messages
    """
    try:
        # Convert dict messages to BaseMessage objects if needed
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_type = msg.get("type", "human")
                content = msg.get("content", "")

                if msg_type == "human":
                    langchain_messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    langchain_messages.append(AIMessage(content=content))
                elif msg_type == "system":
                    langchain_messages.append(SystemMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(msg)

        # Use the Django memory provider to save messages
        chat_history = get_chat_history(session_id, provider="django")
        chat_history.add_messages(langchain_messages)

        logger.info(f"Saved {len(langchain_messages)} messages to session {session_id}")

    except Exception as e:
        logger.error(f"Error saving messages: {e}", exc_info=True)
        raise ChainExecutionError(f"Failed to save messages: {e!s}") from e


# Legacy compatibility functions (deprecated)
def get_in_memory_by_session_id(session_id: Union[str, uuid.UUID]) -> BaseChatMessageHistory:
    """
    DEPRECATED: Use get_chat_history(session_id, provider='inmemory') instead.
    """
    logger.warning("get_in_memory_by_session_id is deprecated. Use get_chat_history() instead.")
    return get_chat_history(session_id, provider="inmemory")


def get_postgres_by_session_id(
    session_id: Union[str, uuid.UUID], **kwargs
) -> BaseChatMessageHistory:
    """
    DEPRECATED: Use get_chat_history(session_id, provider='django') instead.
    """
    logger.warning("get_postgres_by_session_id is deprecated. Use get_chat_history() instead.")
    return get_chat_history(session_id, provider="django")

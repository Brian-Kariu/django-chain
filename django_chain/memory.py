"""
Memory management for django-chain.
"""

import logging
from typing import Any, Optional, Union

from django.apps import apps
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from django_chain.exceptions import ChainExecutionError

logger = logging.getLogger(__name__)


def get_langchain_memory(
    session: Any,
    memory_type: str = "buffer",
    k: Optional[int] = None,
) -> Any:
    """
    Get a LangChain memory object hydrated with messages from the given session.

    Args:
        session: The chat session to get memory for
        memory_type: The type of memory to use ('buffer' or 'buffer_window')
        k: The number of messages to keep in window memory

    Returns:
        A configured LangChain memory object

    Raises:
        ChainExecutionError: If there's an error creating the memory
    """
    try:
        # Get messages from the session
        chat_messages = session.messages.order_by("timestamp", "order").all()
        history = []

        # Convert Django messages to LangChain messages
        for msg in chat_messages:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                history.append(AIMessage(content=msg.content))
            # Add other roles as needed

        # Create the appropriate memory type
        if memory_type == "buffer":
            memory = ConversationBufferMemory(return_messages=True)
        elif memory_type == "buffer_window":
            memory = ConversationBufferWindowMemory(k=k or 5, return_messages=True)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        # Add messages to memory
        for msg in history:
            if isinstance(msg, HumanMessage):
                memory.chat_memory.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                memory.chat_memory.add_ai_message(msg.content)

        return memory

    except Exception as e:
        logger.error(f"Error creating memory: {e}", exc_info=True)
        raise ChainExecutionError(f"Failed to create memory: {e!s}") from e


def save_messages_to_session(
    session: Any,
    messages: list[Union[dict[str, Any], BaseMessage]],
) -> None:
    """
    Save a list of LangChain messages to the chat session.

    Args:
        session: The chat session to save messages to
        messages: The list of LangChain messages or dictionaries to save
    """
    try:
        ChatMessage = apps.get_model("django_chain", "ChatMessage")
        role_map = {
            "human": "user",
            "ai": "ai",
            "system": "system",
            "function": "function",
        }

        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = role_map.get(msg.get("type", "human"), "user")
                content = msg.get("content", "")
            else:
                role = role_map.get(msg.type, "user")  # Default to user if type unknown
                content = msg.content

            ChatMessage.objects.create(
                session=session,
                content=content,
                role=role,
                order=session.messages.count() + i,
            )

        session.save()  # Update updated_at timestamp

    except Exception as e:
        logger.error(f"Error saving messages: {e}", exc_info=True)
        raise ChainExecutionError(f"Failed to save messages: {e!s}") from e

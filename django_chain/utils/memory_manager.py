"""
Memory manager service for handling chat history.
"""

from langchain.memory import PostgresChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pydantic import Field

from django_chain.exceptions import ChainExecutionError

# TODO: CRITICAL!!! Use the messaging model in place of this
store = {}


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> list[BaseMessage]:
        """Get the chat history."""
        return self.messages


def get_in_memory_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


def get_postgres_by_session_id(session_id: str, **kwargs) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = PostgresChatMessageHistory(**kwargs)
    return store[session_id]


def _get_memory_implementation(implementation="INMEMORY"):
    # TODO: This could follow same pattern as providers
    memory_implementation_map = {
        "INMEMORY": get_in_memory_by_session_id,
        "POSTGRES": get_postgres_by_session_id,
    }
    memory_func = memory_implementation_map[implementation]
    return memory_func

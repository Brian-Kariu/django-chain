"""
Examples demonstrating the improved memory manager integration with Django models.

This module shows how to use the refactored memory_manager.py with Django's
ChatHistory model as the primary storage mechanism.
"""

import uuid
from django.contrib.auth import get_user_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from django_chain.models import ChatSession, ChatHistory
from django_chain.utils.memory_manager import (
    get_langchain_memory,
    get_chat_history,
    save_messages_to_session,
    MemoryManager,
)

User = get_user_model()


def example_basic_memory_usage():
    """Basic example of using the Django-backed memory system."""
    print("=== Basic Memory Usage Example ===")

    # Create a chat session
    session = ChatSession.objects.create(
        title="Memory Example Chat",
        user=None,  # Anonymous session
    )
    print(f"Created session: {session.session_id}")

    # Add some messages using the Django model directly
    session.add_message("Hello, I'm a user!", role="USER")
    session.add_message("Hi there! I'm an AI assistant.", role="ASSISTANT")
    session.add_message("What can you help me with?", role="USER")

    print(f"Added 3 messages. Total count: {session.get_message_count()}")

    # Get LangChain memory object
    memory = session.get_memory(memory_type="buffer")
    print(f"Memory has {len(memory.chat_memory.messages)} messages")

    # Use the memory in a conversation
    memory.chat_memory.add_user_message("Tell me about Django")
    memory.chat_memory.add_ai_message("Django is a high-level Python web framework!")

    print(f"After adding via LangChain: {session.get_message_count()} messages in DB")

    return session


def example_window_memory():
    """Example of using window memory with limited message history."""
    print("\n=== Window Memory Example ===")

    session = ChatSession.objects.create(title="Window Memory Test")

    # Add many messages
    for i in range(10):
        session.add_message(f"User message {i + 1}", role="USER")
        session.add_message(f"Assistant response {i + 1}", role="ASSISTANT")

    print(f"Added 20 messages total")

    # Get window memory that only keeps last 6 messages
    window_memory = session.get_memory(memory_type="buffer_window", k=6)
    messages = window_memory.chat_memory.messages

    print(f"Window memory contains {len(messages)} messages (should be 6)")
    for i, msg in enumerate(messages):
        print(f"  {i + 1}. {msg.type}: {msg.content}")

    return session


def example_memory_providers():
    """Example of using different memory providers."""
    print("\n=== Memory Providers Example ===")

    session = ChatSession.objects.create(title="Provider Test")
    session_id = session.session_id

    # Use Django provider (default)
    django_memory = get_langchain_memory(session_id, provider="django")
    django_memory.chat_memory.add_user_message("Message via Django provider")

    # Use legacy in-memory provider for comparison
    inmemory_memory = get_langchain_memory(session_id, provider="inmemory")
    inmemory_memory.chat_memory.add_user_message("Message via in-memory provider")

    # Check what's in the database (only Django provider saves to DB)
    db_count = session.get_message_count()
    print(f"Messages in database: {db_count} (should be 1)")

    # Django provider messages persist
    django_memory2 = get_langchain_memory(session_id, provider="django")
    print(f"Django provider messages: {len(django_memory2.chat_memory.messages)}")

    # In-memory provider messages don't persist between instances
    inmemory_memory2 = get_langchain_memory(session_id, provider="inmemory")
    print(f"In-memory provider messages: {len(inmemory_memory2.chat_memory.messages)}")

    return session


def example_langchain_integration():
    """Example of seamless LangChain integration."""
    print("\n=== LangChain Integration Example ===")

    session = ChatSession.objects.create(title="LangChain Integration")

    # Create LangChain messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
        HumanMessage(content="What about Germany?"),
    ]

    # Save to session using the utility function
    save_messages_to_session(session.session_id, messages)

    print(f"Saved {len(messages)} LangChain messages to database")
    print(f"Database now has {session.get_message_count()} messages")

    # Retrieve as LangChain memory
    memory = session.get_memory()
    retrieved_messages = memory.chat_memory.messages

    print("Retrieved messages:")
    for i, msg in enumerate(retrieved_messages):
        print(f"  {i + 1}. {msg.__class__.__name__}: {msg.content}")

    # Show that we can convert individual messages
    last_message = session.messages.last()
    langchain_msg = last_message.to_langchain_message()
    print(f"\nLast message as LangChain: {langchain_msg.__class__.__name__}")

    return session


def example_custom_provider():
    """Example of registering a custom memory provider."""
    print("\n=== Custom Provider Example ===")

    from django_chain.utils.memory_manager import BaseMemoryProvider
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain.memory import ConversationBufferMemory

    class CustomMemoryProvider(BaseMemoryProvider):
        """Custom provider that adds metadata to messages."""

        def __init__(self):
            self._store = {}

        def get_chat_history(self, session_id):
            session_key = str(session_id)
            if session_key not in self._store:
                self._store[session_key] = ChatMessageHistory()
            return self._store[session_key]

        def get_memory(self, session_id, memory_type="buffer", **kwargs):
            chat_history = self.get_chat_history(session_id)
            return ConversationBufferMemory(
                chat_memory=chat_history, return_messages=True, **kwargs
            )

    # Register the custom provider
    MemoryManager.register_provider("custom", CustomMemoryProvider)

    # Use the custom provider
    session = ChatSession.objects.create(title="Custom Provider Test")
    custom_memory = get_langchain_memory(session.session_id, provider="custom")
    custom_memory.chat_memory.add_user_message("Hello from custom provider!")

    print("Custom provider registered and used successfully")
    print(f"Custom memory has {len(custom_memory.chat_memory.messages)} messages")

    return session


def example_session_management():
    """Example of advanced session management features."""
    print("\n=== Session Management Example ===")

    user = User.objects.create_user(username="testuser", email="test@example.com")
    session = ChatSession.objects.create(title="Advanced Session", user=user)

    # Add messages with metadata
    session.add_message("Hello with metadata", role="USER", token_count=5)
    session.add_message("Response with metadata", role="ASSISTANT", token_count=8)

    print(f"Session has {session.get_message_count()} messages")

    # Demonstrate soft delete
    first_message = session.messages.first()
    first_message.soft_delete()

    print(f"After soft delete: {session.get_message_count()} active messages")
    print(f"Total including deleted: {session.get_message_count(include_deleted=True)} messages")

    # Clear all messages (soft delete)
    session.clear_messages(hard_delete=False)
    print(f"After clearing: {session.get_message_count()} active messages")

    # Memory should be empty now
    memory = session.get_memory()
    print(f"Memory has {len(memory.chat_memory.messages)} messages (should be 0)")

    return session


def run_all_examples():
    """Run all memory manager examples."""
    print("Django-Chain Memory Manager Examples")
    print("=" * 50)

    try:
        example_basic_memory_usage()
        example_window_memory()
        example_memory_providers()
        example_langchain_integration()
        example_custom_provider()
        example_session_management()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # This would be run in a Django shell or management command
    print("To run these examples, use:")
    print("python manage.py shell")
    print(">>> from example.memory_examples import run_all_examples")
    print(">>> run_all_examples()")

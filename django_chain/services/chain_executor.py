"""
Chain execution service for django-chain.
"""

import logging
import time
from typing import Any, Optional

from django.apps import apps
from django.conf import settings

from django_chain.exceptions import ChainExecutionError
from django_chain.memory import get_langchain_memory, save_messages_to_session
from django_chain.models.logs import LLMInteractionLog
from django_chain.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ChainExecutor:
    """
    Service for executing LLM chains.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the chain executor.

        Args:
            llm_client: Optional LLM client to use. If not provided, a default one will be created.
        """
        self.llm_client = llm_client or LLMClient()

    def execute_chain(
        self,
        chain_id: int,
        input_data: dict[str, Any],
        user: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Execute a chain with the given input data.

        Args:
            chain_id: The ID of the chain to execute
            input_data: The input data for the chain
            user: Optional user to associate with the execution
            session: Optional chat session to use for memory

        Returns:
            The chain's output

        Raises:
            ChainExecutionError: If there's an error executing the chain
        """
        try:
            LLMChain = apps.get_model("django_chain", "LLMChain")
            chain = LLMChain.objects.get(id=chain_id)

            langchain_chain = chain.get_chain(self.llm_client)

            if session:
                memory = get_langchain_memory(session)
                langchain_chain.memory = memory

            start_time = time.time()
            output = langchain_chain.run(**input_data)
            end_time = time.time()

            if session and hasattr(langchain_chain, "memory"):
                save_messages_to_session(session, langchain_chain.memory.chat_memory.messages)

            LLMInteractionLog.objects.create(
                user=user,
                prompt_text=str(input_data),
                response_text=str(output),
                model_name=chain.model_name,
                provider=chain.provider,
                input_tokens=0,  # TODO: Implement token counting
                output_tokens=0,  # TODO: Implement token counting
                total_cost=0,  # TODO: Implement cost calculation
                latency_ms=int((end_time - start_time) * 1000),
                status="success",
            )

            return output

        except Exception as e:
            logger.error(f"Error executing chain: {e}", exc_info=True)
            raise ChainExecutionError(f"Failed to execute chain: {e!s}") from e

    @classmethod
    def _log_interaction(
        cls,
        user: Optional[Any],
        prompt_text: str,
        response_text: str,
        model_name: str,
        provider: str,
        latency_ms: int,
        status: str,
        error_message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log an LLM interaction to the database.
        """
        if not settings.DJANGO_LLM_SETTINGS.get("ENABLE_LLM_LOGGING", True):
            return

        try:
            LLMInteractionLog.objects.create(
                user=user,
                prompt_text=prompt_text,
                response_text=response_text,
                model_name=model_name,
                provider=provider,
                latency_ms=latency_ms,
                status=status,
                error_message=error_message,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Failed to log LLM interaction: {e}", exc_info=True)

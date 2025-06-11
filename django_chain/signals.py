"""
Signal definitions for django-chain.
"""

from django.dispatch import Signal

# Signal sent when an LLM interaction is completed
llm_interaction_completed = Signal()
# Arguments: ['model_name', 'input_tokens', 'output_tokens', 'latency_ms', 'cost', 'user', 'success']

# Signal sent when an LLM interaction fails
llm_interaction_failed = Signal()
# Arguments: ['model_name', 'error_message', 'user']

# Signal sent when a chain is executed
chain_executed = Signal()
# Arguments: ['chain_name', 'input_data', 'output_data', 'execution_time', 'user']

# Signal sent when a vector store operation is performed
vector_store_operation = Signal()
# Arguments: ['operation_type', 'document_count', 'success', 'error_message']

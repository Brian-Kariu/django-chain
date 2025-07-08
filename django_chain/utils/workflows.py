import os

from django.db import models
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from django_chain.exceptions import WorkflowValidationError
from django_chain.providers import get_chat_model


class LangchainChainTypes(models.TextChoices):
    Prompt = "Prompt", "prompt"
    Model = "LLM", "llm"
    Parser = "Parser", "parser"


class ParserTypes(models.TextChoices):
    StrOutputParser = "string", "StrOutputParser"
    JsonOutputParser = "json", "JsonOutputParser"


def _evaluate_prompt(step_data, **kwargs):
    prompt_instance = kwargs.get("prompt_instance")
    prompt_object = prompt_instance.to_langchain_prompt()
    return prompt_object


def _evaluate_model(step_data, **config):
    llm_config = config.get("llm_config", [])
    llm_config_override = step_data.get("config", {})
    current_llm_config = {
        **llm_config,
        **llm_config_override,
    }

    llm_provider = current_llm_config.get("DEFAULT_LLM_PROVIDER")
    model_name = current_llm_config["DEFAULT_CHAT_MODEL"]["name"]
    temperature = current_llm_config["DEFAULT_CHAT_MODEL"]["temperature"]
    api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")

    if not llm_provider or not model_name:
        raise ValueError(
            f"Workflow step {i}: LLM step requires 'llm_provider' and 'model_name' in its config or global LLM config."
        )

    llm_instance = get_chat_model(llm_provider, temperature=temperature, api_key=api_key)

    return llm_instance


def _evaluate_parser(step_data, **config):
    parser_types_mapping = {
        "StrOutputParser": StrOutputParser,
        "JsonOutputParser": JsonOutputParser,
    }
    parser_type = step_data.get("parser_type")
    parser_args = step_data.get("parser_args", {})
    parser_instance = None
    parser_instance = parser_types_mapping[parser_type](**parser_args)

    return parser_instance


def _convert_to_runnable(workflow, **kwargs) -> list:
    chain_components = []
    step_data_mapping = {
        "prompt": _evaluate_prompt,
        "llm": _evaluate_model,
        "parser": _evaluate_parser,
    }
    for i, step_data in enumerate(workflow.workflow_definition):
        step_type = step_data.get("type")
        chain_components.append(step_data_mapping[step_type](step_data, **kwargs))

    if len(chain_components) == 0:
        raise WorkflowValidationError(
            chain_components, "Workflow definition is empty or contains no valid components."
        )
    return chain_components

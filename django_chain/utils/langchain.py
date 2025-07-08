import os
from abc import ABC
from abc import abstractmethod
from functools import reduce
from typing import Sequence

from django.db import models
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import AIMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from django_chain.exceptions import PromptValidationError
from django_chain.exceptions import WorkflowValidationError
from django_chain.providers import get_chat_model
from django_chain.utils.llm_client import LoggingHandler
from django_chain.utils.llm_client import _execute_and_log_workflow_step
from django_chain.utils.llm_client import add_wrapper_function


class ParserTypes(models.TextChoices):
    StrOutputParser = "string", "StrOutputParser"
    JsonOutputParser = "json", "JsonOutputParser"


class MessageTypes(models.TextChoices):
    Role = "Role", "role"
    Content = "Content", "content"
    Placeholder = "Placeholder", "placeholder"
    MessagePlaceholder = "MessagesPlaceholder", "MessagesPlaceholder"
    SystemMessagePromptTemplate = "SystemMessagePromptTemplate", "SystemMessagePromptTemplate"
    AIMessagePromptTemplate = "AIMessagePromptTemplate", "AIMessagePromptTemplate"
    HumanMessagePromptTemplate = "HumanMessagePromptTemplate", "HumanMessagePromptTemplate"


class PromptTemplateInstantiation(models.TextChoices):
    Basic = "basic", "Basic"
    Template = "template", "Template"


class LangchainPromptChoices(models.TextChoices):
    PromptTemplate = "Prompt", "PromptTemplate"
    ChatPromptTemplate = "Chat", "ChatPromptTemplate"


class ChainInterface(ABC):
    @abstractmethod
    def get_langchain_object(self) -> BasePromptTemplate | None:
        pass

    def execute(self):
        pass


class PromptObject(ChainInterface):
    def __init__(self, prompt_template: LangchainPromptChoices | None = None):
        self.prompt_template = prompt_template

    def get_langchain_object(self, *args, **kwargs) -> BasePromptTemplate | None:
        """Converts prompt to its langchain equivalent"""
        langchain_object = None
        prompt_template = kwargs.pop("prompt_template") or self.prompt_template
        methods = [
            name
            for name in dir(self)
            if callable(getattr(self, name)) and not name.startswith("__") and "template" in name
        ]
        for method in methods:
            formatted_method = "".join(method.split("_")[2:])
            if str(prompt_template).lower() in formatted_method:
                template_method = getattr(self, method)
                langchain_object = template_method(*args, **kwargs)

        if langchain_object is None:
            raise PromptValidationError(value=prompt_template, expected_format="a prompt template")

        return langchain_object

    def _get_prompt_template(
        self,
        template: str,
        instantiation_method: PromptTemplateInstantiation = PromptTemplateInstantiation.Basic,
        validate_tempate: bool = True,
        *args,
        **kwargs,
    ) -> PromptTemplate | None:
        prompt_template = None
        if instantiation_method == PromptTemplateInstantiation.Basic.value:
            prompt_template = PromptTemplate(
                template=template, validate_template=validate_tempate, *args, **kwargs
            )
        if instantiation_method == PromptTemplateInstantiation.Template.value:
            prompt_template = PromptTemplate.from_template(
                validate_tempate=validate_tempate, *args, **kwargs
            )
        return prompt_template

    def _get_chat_prompt_template(
        self,
        instantiation_method: PromptTemplateInstantiation = PromptTemplateInstantiation.Basic,
        validate_template: bool = True,
        *args,
        **kwargs,
    ) -> ChatPromptTemplate | None:
        prompt_template = None
        messages = []
        for key, value in kwargs.items():
            if key == "messages":
                messages.extend(self._get_template_messages(value))
        if len(messages) >= 0:
            kwargs.pop("messages")
        if instantiation_method == PromptTemplateInstantiation.Basic.value:
            prompt_template = ChatPromptTemplate(
                validate_template=validate_template, messages=messages, *args, **kwargs
            )
        if instantiation_method == PromptTemplateInstantiation.Template.value:
            prompt_template = ChatPromptTemplate.from_template(
                validate_template=validate_template, messages=messages, *args, **kwargs
            )
        return prompt_template

    def _get_template_messages(self, messages: Sequence[dict]) -> Sequence[BaseMessage | dict]:
        message_list = []
        current_role = ""
        for message in messages:
            for key, value in message.items():
                if key == MessageTypes.Role.label:
                    current_role = value
                if key == MessageTypes.Content.label:
                    message_list.append((current_role, value))
                if key == MessageTypes.MessagePlaceholder.label:
                    message_list.append(MessagesPlaceholder(value))
                if key == MessageTypes.HumanMessagePromptTemplate.label:
                    message_list.append(HumanMessagePromptTemplate.from_template(value))
                if key == MessageTypes.SystemMessagePromptTemplate.label:
                    message_list.append(SystemMessagePromptTemplate.from_template(value))
                if key == MessageTypes.AIMessagePromptTemplate.label:
                    message_list.append(AIMessagePromptTemplate.from_template(value))
        return message_list

    def execute(self):
        """Creates and saves an instance of a prompt"""
        return NotImplemented


class Workflow(ChainInterface):
    def __init__(self) -> None:
        pass

    def get_langchain_object(self, *args, **kwargs) -> BasePromptTemplate | None:
        """Converts workflow to its langchain chain equivalent"""
        langchain_object = None
        step_data_values = ["prompt", "llm", "parser"]
        workflow_definition = kwargs.get("workflow_definition")
        for i, step_data in enumerate(workflow_definition):
            methods = [
                name
                for name in dir(self)
                if callable(getattr(self, name))
                and not name.startswith("__")
                and name in step_data_values
            ]
            for method in methods:
                step_type = step_data.pop("type")
                if str(step_type).lower() in method:
                    template_method = getattr(self, method)
                    langchain_object = template_method(**step_data)

            if langchain_object is None:
                raise WorkflowValidationError(
                    value=step_data, expected_format="a workflow_defintion template"
                )

        return langchain_object

    def _evaluate_parser(self, step_data, **config):
        parser_types_mapping = {
            "StrOutputParser": StrOutputParser,
            "JsonOutputParser": JsonOutputParser,
        }
        parser_type = step_data.get("parser_type")
        parser_args = step_data.get("parser_args", {})
        parser_instance = None
        parser_instance = parser_types_mapping[parser_type](**parser_args)

        return parser_instance

    def _evaluate_prompt(self, **kwargs):
        prompt_instance = kwargs.get("prompt_instance")
        prompt_object = prompt_instance.to_langchain_prompt()
        return prompt_object

    def _evaluate_model(self, **kwargs):
        step_data = kwargs.get("step_data")
        llm_config = kwargs.get("llm_config", [])
        llm_config_override = step_data.get("config", {})
        current_llm_config = {
            **llm_config,
            **llm_config_override,
        }

        llm_provider = current_llm_config.get("DEFAULT_LLM_PROVIDER", "")
        model_name = current_llm_config["DEFAULT_CHAT_MODEL"]["name"]
        temperature = current_llm_config["DEFAULT_CHAT_MODEL"]["temperature"]
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")

        if llm_provider == "" or not model_name:
            raise WorkflowValidationError(
                value=step_data,
                expected_format="a workflow definition step",
                additional_message=f"LLM step requires 'llm_provider' and 'model_name' in its config or global LLM config.",
            )

        llm_instance = get_chat_model(llm_provider, temperature=temperature, api_key=api_key)

        return llm_instance

    def execute(self, **kwargs):
        chain_components = kwargs.get("chains")
        input_data = kwargs.get("input_data")
        execution_method = kwargs.get("execution_method")
        execution_config = kwargs.get("execution_config")
        session = kwargs.get("session_id")
        interaction_log = kwargs.get("interaction_log")
        handlers = []
        if chain_components is None:
            raise WorkflowValidationError(
                value=chain_components, expected_format="a langchain component"
            )
        if interaction_log:
            handlers.append(LoggingHandler(interaction_log=interaction_log))
            workflow_chain = reduce(lambda a, b: a | b, chain_components).with_config(
                callbacks=handlers
            )
        else:
            workflow_chain = reduce(lambda a, b: a | b, chain_components)

        if session:
            input_messages_key = kwargs.get("chat_input")
            history = kwargs.get("history")
            workflow_chain = add_wrapper_function(
                chain=workflow_chain,
                function_name="runnable_with_message_history",
                input_messages_key=input_messages_key,
                history_messages_key=history,
            )
        response = _execute_and_log_workflow_step(
            workflow_chain, input_data, execution_method, execution_config
        )
        return response

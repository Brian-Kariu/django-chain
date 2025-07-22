import os
from functools import reduce
from typing import Any
from typing import List
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


class LangchainChainTypes(models.TextChoices):
    Prompt = "Prompt", "prompt"
    Model = "LLM", "llm"
    Parser = "Parser", "parser"


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


class PromptObject:
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
            if str(prompt_template).lower() == formatted_method.lower():
                template_method = getattr(self, method)
                langchain_object = template_method(*args, **kwargs)
                break

        if langchain_object is None:
            raise PromptValidationError(value=prompt_template, expected_format="a prompt template")

        return langchain_object

    def _get_prompt_template(
        self,
        template: str | None = "",
        instantiation_method: PromptTemplateInstantiation = PromptTemplateInstantiation.Basic,
        validate_tempate: bool = True,
        *args,
        **kwargs,
    ) -> PromptTemplate | None:
        if template == "":
            current_template = self.prompt_template or None
        else:
            current_template = template
        if instantiation_method == PromptTemplateInstantiation.Basic.value:
            prompt_template = PromptTemplate(
                template=current_template, validate_template=validate_tempate, *args, **kwargs
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
        if len(messages) >= 0 and "messages" in kwargs:
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


class Workflow:
    """
    Dynamic Workflow processor that can convert workflow definitions to LangChain components.

    This class provides a standalone way to process workflow definitions without being tied
    to Django models, providing better error handling, extensibility, and maintainability.
    """

    def __init__(self) -> None:
        """Initialize the workflow processor with step type mappings."""
        self.step_data_mapping = {
            "prompt": self._evaluate_prompt,
            "llm": self._evaluate_model,
            "parser": self._evaluate_parser,
        }

    def convert_to_runnable_components(self, workflow_definition, **kwargs) -> List[Any]:
        """
        Convert workflow definition to a list of LangChain runnable components.

        Args:
            workflow_definition: List of step dictionaries defining the workflow
            **kwargs: Additional configuration parameters

        Returns:
            List of LangChain runnable components

        Raises:
            WorkflowValidationError: If workflow definition is invalid or empty
        """
        if not workflow_definition:
            raise WorkflowValidationError(
                workflow_definition, "Workflow definition is empty or None."
            )
        chain_components = []

        for i, step_data in enumerate(workflow_definition):
            try:
                step_type = step_data.get("type")
                if not step_type:
                    raise WorkflowValidationError(
                        step_data, f"Workflow step {i}: Missing 'type' field in step definition."
                    )

                if step_type not in self.step_data_mapping:
                    raise WorkflowValidationError(
                        step_data,
                        f"Workflow step {i}: Unsupported step type '{step_type}'. "
                        f"Supported types: {list(self.step_data_mapping.keys())}",
                    )

                # Get the evaluation function for this step type
                evaluation_function = self.step_data_mapping[step_type]

                # Evaluate the step and get the component
                component = evaluation_function(step_data, step_index=i, **kwargs)
                chain_components.append(component)

            except Exception as e:
                if isinstance(e, WorkflowValidationError):
                    raise
                else:
                    raise WorkflowValidationError(
                        step_data, f"Workflow step {i}: Error processing step - {str(e)}"
                    ) from e

        if len(chain_components) == 0:
            raise WorkflowValidationError(
                workflow_definition,
                "Workflow definition contains no valid components after processing.",
            )

        return chain_components

    def _evaluate_prompt(self, step_data: dict, step_index: int = 0, **kwargs) -> Any:
        """
        Evaluate a prompt step and return the LangChain prompt object.

        Args:
            step_data: The step configuration
            step_index: Index of the step in the workflow
            **kwargs: Additional configuration

        Returns:
            LangChain prompt object

        Raises:
            WorkflowValidationError: If prompt evaluation fails
        """
        try:
            prompt_instance = kwargs.get("prompt_instance")

            # If no prompt_instance provided, try to look up by name
            if not prompt_instance:
                prompt_name = step_data.get("name")
                if prompt_name:
                    # Import here to avoid circular imports
                    from django_chain.models import Prompt

                    try:
                        prompt_instance = Prompt.objects.get(name=prompt_name, is_active=True)
                    except Prompt.DoesNotExist:
                        raise WorkflowValidationError(
                            step_data,
                            f"Workflow step {step_index}: Prompt with name '{prompt_name}' not found or not active.",
                        )
                else:
                    raise WorkflowValidationError(
                        step_data,
                        f"Workflow step {step_index}: No prompt_instance provided and no 'name' specified in step data.",
                    )

            prompt_object = prompt_instance.to_langchain_prompt()
            if not prompt_object:
                raise WorkflowValidationError(
                    step_data,
                    f"Workflow step {step_index}: Failed to convert prompt to LangChain object.",
                )

            return prompt_object

        except Exception as e:
            if isinstance(e, WorkflowValidationError):
                raise
            else:
                raise WorkflowValidationError(
                    step_data, f"Workflow step {step_index}: Error evaluating prompt - {str(e)}"
                ) from e

    def _evaluate_model(self, step_data: dict, step_index: int = 0, **kwargs) -> Any:
        """
        Evaluate an LLM step and return the LangChain model object.

        Args:
            step_data: The step configuration
            step_index: Index of the step in the workflow
            **kwargs: Additional configuration

        Returns:
            LangChain model object

        Raises:
            WorkflowValidationError: If model evaluation fails
        """
        try:
            llm_config = kwargs.get("llm_config", {})
            llm_config_override = step_data.get("config", {})
            current_llm_config = {
                **llm_config,
                **llm_config_override,
            }

            llm_provider = current_llm_config.get("DEFAULT_LLM_PROVIDER")
            if not llm_provider:
                raise WorkflowValidationError(
                    step_data,
                    f"Workflow step {step_index}: Missing 'DEFAULT_LLM_PROVIDER' in LLM configuration.",
                )

            chat_model_config = current_llm_config.get("DEFAULT_CHAT_MODEL", {})
            if not chat_model_config:
                raise WorkflowValidationError(
                    step_data,
                    f"Workflow step {step_index}: Missing 'DEFAULT_CHAT_MODEL' in LLM configuration.",
                )

            model_name = chat_model_config.get("name")
            temperature = chat_model_config.get("temperature", 0.7)

            if not model_name:
                raise WorkflowValidationError(
                    step_data,
                    f"Workflow step {step_index}: Missing 'name' in DEFAULT_CHAT_MODEL configuration.",
                )

            # Get API key from environment
            api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
            if not api_key and llm_provider != "fake":
                # For non-fake providers, we might want to warn but not fail
                # since the provider might handle API keys differently
                pass

            llm_instance = get_chat_model(
                llm_provider, temperature=temperature, api_key=api_key, model=model_name
            )

            return llm_instance

        except Exception as e:
            if isinstance(e, WorkflowValidationError):
                raise
            else:
                raise WorkflowValidationError(
                    step_data, f"Workflow step {step_index}: Error evaluating LLM model - {str(e)}"
                ) from e

    def _evaluate_parser(self, step_data: dict, step_index: int = 0, **kwargs) -> Any:
        """
        Evaluate a parser step and return the LangChain parser object.

        Args:
            step_data: The step configuration
            step_index: Index of the step in the workflow
            **kwargs: Additional configuration

        Returns:
            LangChain parser object

        Raises:
            WorkflowValidationError: If parser evaluation fails
        """
        try:
            parser_types_mapping = {
                "StrOutputParser": StrOutputParser,
                "JsonOutputParser": JsonOutputParser,
            }

            parser_type = step_data.get("parser_type")
            if not parser_type:
                raise WorkflowValidationError(
                    step_data, f"Workflow step {step_index}: Missing 'parser_type' in parser step."
                )

            if parser_type not in parser_types_mapping:
                raise WorkflowValidationError(
                    step_data,
                    f"Workflow step {step_index}: Unsupported parser type '{parser_type}'. "
                    f"Supported types: {list(parser_types_mapping.keys())}",
                )

            parser_args = step_data.get("parser_args", {})
            parser_class = parser_types_mapping[parser_type]
            parser_instance = parser_class(**parser_args)

            return parser_instance

        except Exception as e:
            if isinstance(e, WorkflowValidationError):
                raise
            else:
                raise WorkflowValidationError(
                    step_data, f"Workflow step {step_index}: Error evaluating parser - {str(e)}"
                ) from e

    def get_langchain_object(self, *args, **kwargs) -> BasePromptTemplate | None:
        """
        Converts workflow to its langchain chain equivalent.

        This method is kept for compatibility with the ChainInterface,
        but the main functionality is now in convert_to_runnable_components.
        """
        workflow_definition = kwargs.get("workflow_definition")
        if not workflow_definition:
            raise WorkflowValidationError(
                None, "No workflow_definition provided to get_langchain_object"
            )

        components = self.convert_to_runnable_components(workflow_definition, **kwargs)

        # Return the first component if there's only one, otherwise return None
        # This method is mainly for interface compatibility
        return components[0] if len(components) == 1 else None

    def execute(self, workflow_definition, **kwargs):
        """
        Execute a workflow with the given parameters.

        This method provides a complete workflow execution with improved error handling
        and configuration management.

        Args:
            workflow_definition: List of step dictionaries defining the workflow
            **kwargs: Additional execution parameters including:
                - input_data: Input data for the workflow
                - execution_method: Method to use for execution (default: "invoke")
                - execution_config: Configuration for execution
                - session_id: Session ID for chat history
                - interaction_log: Log instance for tracking
                - llm_config: LLM configuration
                - chat_input: Chat input key for message history
                - history: History key for message history

        Returns:
            The result of workflow execution
        """
        # Convert workflow definition to runnable components
        chain_components = self.convert_to_runnable_components(workflow_definition, **kwargs)

        input_data = kwargs.get("input_data")
        execution_method = kwargs.get("execution_method", "invoke")
        execution_config = kwargs.get("execution_config", {})
        session = kwargs.get("session_id")
        interaction_log = kwargs.get("interaction_log")

        if not chain_components:
            raise WorkflowValidationError(
                None, "No chain components generated from workflow definition"
            )

        # Set up logging handlers if interaction log is provided
        handlers = []
        if interaction_log:
            handlers.append(LoggingHandler(interaction_log=interaction_log))
            workflow_chain = reduce(lambda a, b: a | b, chain_components).with_config(
                callbacks=handlers
            )
        else:
            workflow_chain = reduce(lambda a, b: a | b, chain_components)

        # Add message history wrapper if session is provided
        if session:
            input_messages_key = kwargs.get("chat_input")
            history = kwargs.get("history")
            workflow_chain = add_wrapper_function(
                chain=workflow_chain,
                function_name="runnable_with_message_history",
                input_messages_key=input_messages_key,
                history_messages_key=history,
            )

        # Execute the workflow
        response = _execute_and_log_workflow_step(
            workflow_chain, input_data, execution_method, execution_config
        )
        return response

    def register_step_type(self, step_type: str, evaluation_function):
        """
        Register a new step type with its evaluation function.

        This allows extending the workflow processor with custom step types.

        Args:
            step_type: The name of the step type
            evaluation_function: Function that evaluates this step type
        """
        self.step_data_mapping[step_type] = evaluation_function

    def get_supported_step_types(self) -> List[str]:
        """
        Get a list of all supported step types.

        Returns:
            List of supported step type names
        """
        return list(self.step_data_mapping.keys())

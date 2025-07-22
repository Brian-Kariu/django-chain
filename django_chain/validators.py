from operator import add

from django_chain.exceptions import PromptValidationError
from django_chain.exceptions import WorkflowValidationError
from django_chain.utils.langchain import LangchainChainTypes
from django_chain.utils.langchain import LangchainPromptChoices
from django_chain.utils.langchain import ParserTypes


def validate_prompt(prompt_data):
    if not isinstance(prompt_data, dict):
        raise PromptValidationError(
            prompt_data,
            message="The prompt data needs to be a valid dictionary. ",
            expected_format="a dict",
        )
    prompt_types_to_expected_data_map = {
        LangchainPromptChoices.PromptTemplate.name: [
            "template",
        ],
        LangchainPromptChoices.ChatPromptTemplate.name: [
            "messages",
        ],
    }
    template_type = prompt_data.get("langchain_type")
    if template_type is None:
        msg = "'langchain type' key is missing."
        raise PromptValidationError(
            value=template_type, expected_format="a string", additional_message=msg
        )
    if template_type not in [choice[1] for choice in LangchainPromptChoices.choices]:
        msg = (
            f"Supplied langchain prompt type {template_type} is not one of the supported types, "
            f"{' '.join([choice[1] for choice in LangchainPromptChoices.choices])}"
        )
        raise PromptValidationError(
            value=template_type, expected_format="a string", additional_message=msg
        )

    for arg in prompt_types_to_expected_data_map[template_type]:
        current_arg = prompt_data.get(arg)
        if arg == "messages" and current_arg is None:
            msg = f"'messages' key is missing which is required for ChatPromptTemplate."
            raise PromptValidationError(
                value=current_arg, expected_format="a string", additional_message=msg
            )


def validate_workflow(workflow_definition):
    workflow_steps_to_expected_validation_map = {
        LangchainChainTypes.Prompt.label: ["name"],
        LangchainChainTypes.Model.label: ["config"],
        LangchainChainTypes.Parser.label: ["parser_type", "parser_args"],
    }

    if workflow_definition is None:
        msg = "'workflow_definition' key is missing"
        WorkflowValidationError(
            value=workflow_definition.workflow_definition, additional_message=msg
        )

    if not isinstance(workflow_definition, list):
        msg = "Invalid_workflow_definition_format"
        raise WorkflowValidationError(
            value=workflow_definition,
            expected_format="JSON array (list of steps).",
            additional_message=msg,
        )

    for i, step in enumerate(workflow_definition):
        if not isinstance(step, dict):
            msg = "Invalid_step_format_{i}"
            raise WorkflowValidationError(
                value=step,
                expected_format="Each step in the workflow definition must be a JSON object.",
                additional_message=msg,
            )

        if step.get("type") not in [choice[1] for choice in LangchainChainTypes.choices]:
            msg = (
                f"Supplied workflow definition type {workflow_definition} "
                f"is not one of the supported types, {' '.join([choice[1] for choice in LangchainChainTypes.choices])}"
            )
            raise WorkflowValidationError(value=workflow_definition, additional_message=msg)

        for arg in workflow_steps_to_expected_validation_map[step.get("type")]:
            if arg == "name" and step.get(arg) is None:
                msg = "'name' not specified"
                WorkflowValidationError(
                    value=step.get(arg), expected_format="string", additional_message=msg
                )
            if arg == "parser_type" and step.get(arg) is None:
                msg = "'parser_type' not specified."
                WorkflowValidationError(
                    value=step.get(arg), expected_format="string", additional_message=msg
                )
            if arg == "parser_type" and step.get(arg) not in [
                choice[1] for choice in ParserTypes.choices
            ]:
                msg = (
                    f"Supplied parser type {arg} is not one of the supported types, "
                    f"{' '.join([choice[1] for choice in ParserTypes.choices])}"
                )
                raise WorkflowValidationError(value=step.get(arg), expected_format=msg)

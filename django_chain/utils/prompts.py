from django.db import models
from langchain_core.prompts import AIMessagePromptTemplate
from langchain_core.prompts import BaseChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate

from django_chain.exceptions import PromptValidationError


class LangchainPromptChoices(models.TextChoices):
    PromptTemplate = "Prompt", "PromptTemplate"
    ChatPromptTemplate = "Chat", "ChatPromptTemplate"


def _create_langchain_template(prompt_type: LangchainPromptChoices, **kwargs):
    # TODO: Rethink everything here!!!
    template = BaseChatPromptTemplate

    if prompt_type == LangchainPromptChoices.PromptTemplate.label:
        template = PromptTemplate(template=prompt_type, **kwargs)

    if prompt_type == LangchainPromptChoices.ChatPromptTemplate.label:
        message_type_mapping = {
            "system": SystemMessagePromptTemplate,
            "placeholder": MessagesPlaceholder,
            "human": HumanMessagePromptTemplate,
            "ai": AIMessagePromptTemplate,
        }
        messages_data = kwargs.get("messages", [])

        langchain_messages = []
        for msg_data in messages_data:
            message_type = msg_data.get("message_type")
            template = msg_data.get("template")
            msg_input_variables = msg_data.get("input_variables", [])

            if template is None:
                raise PromptValidationError(
                    LangchainPromptChoices.ChatPromptTemplate,
                    "a dictionary",
                    f"Chat message of type '{message_type}' requires a 'template' key.",
                )
            if not isinstance(msg_input_variables, list):
                raise PromptValidationError(
                    LangchainPromptChoices.ChatPromptTemplate,
                    "a dictionary",
                    f"input_variables for chat message of type '{message_type}' must be a list.",
                )
            if message_type in message_type_mapping["placeholder"]:
                langchain_messages.append(
                    message_type_mapping[message_type](variable_name=template)
                )

            if message_type in message_type_mapping:
                langchain_messages.append(
                    message_type_mapping[message_type].from_template(template, **kwargs)
                )

            else:
                raise PromptValidationError(
                    LangchainPromptChoices.ChatPromptTemplate,
                    f"Unsupported chat message type: {message_type}",
                )

        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
    return template


def _convert_to_prompt_template(prompt: dict[str, str | list]):
    prompt_data = prompt.prompt_template

    template_args = ["template", "input_variables", "message"]
    template_dict = {
        arg: prompt_data.get(arg) for arg in template_args if prompt_data.get(arg) is not None
    }

    langchain_type = prompt_data.get("langchain_type")
    template = _create_langchain_template(langchain_type, **template_dict)
    return template

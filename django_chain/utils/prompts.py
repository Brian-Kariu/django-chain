from django.db import models

from django_chain.exceptions import PromptValidationError
from django_chain.utils.langchain import PromptObject


def _convert_to_prompt_template(prompt):
    prompt_data = prompt.prompt_template
    if prompt_data is None:
        raise PromptValidationError(value=None, expected_format="a template")

    template_args = ["template", "input_variables", "messages"]
    template_dict = {
        arg: prompt_data.get(arg) for arg in template_args if prompt_data.get(arg) is not None
    }

    langchain_type = prompt_data.get("langchain_type")
    template = PromptObject().get_langchain_object(prompt_template=langchain_type, **template_dict)
    return template

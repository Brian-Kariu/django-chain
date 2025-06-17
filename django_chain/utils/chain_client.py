from typing import Any

from langchain.chains.base import Chain
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate as LC_PromptTemplate


def chat_workflow(
    prompt: LC_PromptTemplate | str, llm: BaseChatModel, parser: BaseOutputParser | None = None
) -> Any:
    if parser is None:
        return prompt | llm
    workflow_chain = prompt | llm | parser
    return workflow_chain

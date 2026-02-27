import operator
from typing import Any

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    used_tools: Annotated[list[str], operator.add]


class AgentOutput(BaseModel):
    answer: str = Field(description="Final answer to the user in portuguese.")
    success: bool = Field(description="Indicates if the agent successfully completed the task.")
    used_tools: list[str] = Field(
        description="List of tools used during the agent's execution.",
        default_factory=list,
    )

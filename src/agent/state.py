import operator
from typing import Any

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class AgentOutput(BaseModel):
    answer: str = Field(description="Final answer to the user in portuguese.")
    success: bool = Field(description="Indicates if the agent successfully completed the task.")

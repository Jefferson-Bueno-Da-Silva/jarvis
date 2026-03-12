import operator
from typing import Any

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


class ToolUsage(TypedDict, total=False):
    name: str
    args: dict[str, Any] | None
    ok: bool | None
    error: str | None


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tools_used: Annotated[list[ToolUsage], operator.add]


class AgentOutput(BaseModel):
    answer: str = Field(description="Final answer to the user in portuguese.")
    success: bool = Field(description="Indicates if the agent successfully completed the task.")
    tools_used: list[ToolUsage] = Field(description="Tools used during the run.")

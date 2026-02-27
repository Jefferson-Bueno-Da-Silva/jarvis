from typing import Literal

from langchain.messages import AIMessage
from langfuse import observe

from src.agent.state import MessagesState


@observe(name="Should Continue?")
def should_continue(state: MessagesState) -> Literal["tool_node", "finalize_node"]:
    """Decide if we should continue the loop or stop based on tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "finalize_node"


@observe(name="Finalize Node")
def finalize_node(state: MessagesState) -> MessagesState:
    final_message = AIMessage(
        content=state.get("messages", "")[-1].content if state.get("messages") else "No messages",
        tool_calls=[],
    )
    return {
        "messages": [final_message],
        "llm_calls": state.get("llm_calls", 0),
        "used_tools": [],
    }

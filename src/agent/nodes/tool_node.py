import json

from langchain.messages import AIMessage, AnyMessage, ToolMessage
from langfuse import observe

from src.agent.state import MessagesState
from src.tools.tools import tools_by_name

@observe(name="Tool Call")
def tool_node(state: MessagesState) -> MessagesState:
    """Performs the tool call decided by the LLM."""
    last_message = state["messages"][-1]

    # Only AIMessage has tool_calls; guard access to satisfy type checkers.
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        empty_messages: list[AnyMessage] = []
        return {
            "messages": empty_messages,
            "llm_calls": state.get("llm_calls", 0),
            "tools_used": [],
        }

    new_messages: list[AnyMessage] = []
    tool_usages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)
        if selected_tool is None:
            tool_output = {"ok": False, "error": f"Ferramenta não encontrada: {tool_name}"}
            tool_usages.append(
                {
                    "name": tool_name,
                    "args": tool_call.get("args"),
                    "ok": False,
                    "error": tool_output.get("error"),
                }
            )
        else:
            tool_output = selected_tool.invoke(tool_call["args"])
            tool_usages.append(
                {
                    "name": tool_name,
                    "args": tool_call.get("args"),
                    "ok": bool(tool_output.get("ok")) if isinstance(tool_output, dict) else None,
                    "error": tool_output.get("error") if isinstance(tool_output, dict) else None,
                }
            )

        tool_message: AnyMessage = ToolMessage(
            content=json.dumps(tool_output, ensure_ascii=False),
            tool_call_id=tool_call["id"],
            name=tool_name,
        )
        new_messages.append(tool_message)

    return {
        "messages": new_messages,
        "llm_calls": state.get("llm_calls", 0),
        "tools_used": tool_usages,
    }

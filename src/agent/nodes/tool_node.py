import json

from langchain.messages import AIMessage, ToolMessage
from langfuse import observe

from src.agent.state import MessagesState
from src.models.model import tools_by_name


@observe(name="Tool Call")
def tool_node(state: MessagesState) -> MessagesState:
    """Performs the tool call decided by the LLM."""
    result = state.get("messages", [])
    used_tools: list[str] = state.get("used_tools", [])
    last_message = state["messages"][-1]

    # Only AIMessage has tool_calls; guard access to satisfy type checkers.
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        return {
            "messages": result,
            "used_tools": used_tools,
            "llm_calls": state.get("llm_calls", 0),
        }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)
        used_tools.append(tool_name)
        if selected_tool is None:
            tool_output = {"ok": False, "error": f"Ferramenta não encontrada: {tool_name}"}
        else:
            tool_output = selected_tool.invoke(tool_call["args"])

        result.append(
            ToolMessage(
                content=json.dumps(tool_output, ensure_ascii=False),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {
        "messages": result,
        "used_tools": used_tools,
        "llm_calls": state.get("llm_calls", 0),
    }

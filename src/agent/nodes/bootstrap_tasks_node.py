import json

from langchain.messages import AnyMessage, SystemMessage
from langfuse import observe

from src.agent.state import MessagesState
from src.tools.tools import tools_by_name


@observe(name="Bootstrap Tasks Node")
def bootstrap_tasks_node(state: MessagesState) -> MessagesState:
    """Preload google_tasks_list output into state before the first LLM call."""
    list_tool = tools_by_name.get("google_tasks_list")
    if list_tool is None:
        empty_messages: list[AnyMessage] = []
        return {
            "messages": empty_messages,
            "llm_calls": state.get("llm_calls", 0),
            "tools_used": [],
        }

    tool_usages = []
    try:
        list_result = list_tool.invoke({"limit": 20})
        tool_usages.append(
            {
                "name": "google_tasks_list",
                "args": {"limit": 20},
                "ok": bool(list_result.get("ok")) if isinstance(list_result, dict) else None,
                "error": list_result.get("error") if isinstance(list_result, dict) else None,
            }
        )
        message = SystemMessage(
            content=(
                "Initial context from google_tasks_list (fetched before first model call). "
                "Use this snapshot first; call google_tasks_list again only if needed.\n"
                f"{json.dumps(list_result, ensure_ascii=False)}"
            )
        )
    except Exception as error:
        tool_usages.append(
            {
                "name": "google_tasks_list",
                "args": {"limit": 20},
                "ok": False,
                "error": str(error),
            }
        )
        message = SystemMessage(
            content=f"Initial google_tasks_list preload failed. Continue without preload. Error: {error}"
        )

    messages: list[AnyMessage] = [message]
    return {
        "messages": messages,
        "llm_calls": state.get("llm_calls", 0),
        "tools_used": tool_usages,
    }

import json

from langchain.messages import SystemMessage
from langfuse import observe

from src.agent.state import MessagesState
from src.models.model import tools_by_name


@observe(name="Bootstrap Tasks Node")
def bootstrap_tasks_node(state: MessagesState) -> MessagesState:
    """Preload google_tasks_list output into state before the first LLM call."""
    result = state.get("messages", [])
    list_tool = tools_by_name.get("google_tasks_list")
    if list_tool is None:
        return {
            "messages": result,
            "used_tools": [],
            "llm_calls": state.get("llm_calls", 0),
        }

    try:
        list_result = list_tool.invoke({"limit": 20})
        result.append(
            SystemMessage(
                content=(
                    "Initial context from google_tasks_list (fetched before first model call). "
                    "Use this snapshot first; call google_tasks_list again only if needed.\n"
                    f"{json.dumps(list_result, ensure_ascii=False)}"
                )
            )
        )
    except Exception as error:
        result.append(
            SystemMessage(
                content=f"Initial google_tasks_list preload failed. Continue without preload. Error: {error}"
            )
        )

    return {
        "messages": result,
        "used_tools": ["google_tasks_list"],
        "llm_calls": state.get("llm_calls", 0),
    }

from collections import Counter

from langchain.messages import AnyMessage, SystemMessage
from langfuse import observe

from src.agent.config import SYS_PROMPT, langfuse, model_with_tools
from src.agent.state import MessagesState


def _tools_used_context(state: MessagesState) -> SystemMessage | None:
    tools_used = state.get("tools_used") or []
    if not tools_used:
        return None

    counts = Counter(
        usage.get("name", "unknown") for usage in tools_used if isinstance(usage, dict)
    )
    summary = ", ".join(f"{name} ({count}x)" for name, count in counts.most_common())
    return SystemMessage(content=f"Tools used so far: {summary}.")


@observe(name="LLM Call")
def llm_call(state: MessagesState) -> MessagesState:
    """LLM decides whether to call a tool or not."""
    tools_context = _tools_used_context(state)
    model_input = [SYS_PROMPT] + ([tools_context] if tools_context else []) + state["messages"]
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="llm-response",
        model="openrouter/free",
        input=model_input,
    ) as generation:
        message = model_with_tools.invoke(model_input)
        generation.update(output=message, metadata=message.response_metadata)

    messages: list[AnyMessage] = [message]
    return {
        "messages": messages,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "tools_used": [],
    }

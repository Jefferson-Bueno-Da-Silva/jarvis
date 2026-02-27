from langfuse import observe

from src.agent.config import SYS_PROMPT, langfuse, model_with_tools
from src.agent.state import MessagesState


@observe(name="LLM Call")
def llm_call(state: MessagesState) -> MessagesState:
    """LLM decides whether to call a tool or not."""
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="llm-response",
        model="gemini-2.5-flash",
        input=[SYS_PROMPT] + state["messages"],
    ) as generation:
        message = model_with_tools.invoke([SYS_PROMPT] + state["messages"])
        generation.update(output=message, metadata=message.response_metadata)

    return {
        "messages": [message],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "used_tools": [],
    }

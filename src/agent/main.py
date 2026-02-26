import json
import operator
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langfuse import Langfuse, observe
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from src.models.model import GeminiModel, tools_by_name
from src.tools.tools import GOOGLE_TASKS_TOOLS

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

model = GeminiModel
model_with_tools = model.bind_tools(GOOGLE_TASKS_TOOLS)

SYS_PROMPT = SystemMessage(
    content=[
        "You are a helpful assistant that can call tools to get information.",
        "translate the response to Portuguese.",
        "To obtain the task ID, you can first use a tool to list all tasks and then retrieve the task ID to execute what was requested.",
    ]
)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


@observe(name="LLM Call")
def llm_call(state: dict) -> dict:
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
    }


@observe(name="Tool Call")
def tool_node(state: dict) -> dict:
    """Performs the tool call decided by the LLM."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)
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
    return {"messages": result}


@observe(name="Should Continue?")
def should_continue(state: MessagesState) -> Literal["tool_node", END]:  # type: ignore
    """Decide if we should continue the loop or stop based on tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)  # type: ignore
agent_builder.add_node("tool_node", tool_node)  # type: ignore

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()


def run_pipeline(user_input: str) -> dict:
    final_state = agent.invoke({"messages": [HumanMessage(content=user_input)]})  # type: ignore
    langfuse.flush()
    return final_state


def extract_final_answer(final_state: dict) -> str:
    for message in reversed(final_state.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(part) for part in content)
            return str(content)
    return "Sem resposta final do agente."

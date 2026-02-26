import json
import operator
import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langfuse import Langfuse, observe
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from src.models.model import GeminiModel, tools_by_name
from src.tools.tools import GOOGLE_TASKS_TOOLS

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    used_tools: Annotated[list[str], operator.add]

class AgentOutput(BaseModel):
    answer: str = Field(description="Final answer to the user in portuguese.")
    success: bool = Field(description="Indicates if the agent successfully completed the task.")
    used_tools: list[str] = Field(description="List of tools used during the agent's execution.", default_factory=list)



model = GeminiModel
model_with_tools = model.bind_tools(GOOGLE_TASKS_TOOLS)
structured_output_model = model.with_structured_output(AgentOutput)

SYS_PROMPT = SystemMessage(
    content=[
        "You are a helpful assistant that can call tools to get information.",
        "translate the response to Portuguese.",
        "To obtain the task ID, you can first use a tool to list all tasks and then retrieve the task ID to execute what was requested.",
    ]
)

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


@observe(name="Should Continue?")
def should_continue(state: MessagesState) -> Literal["tool_node", "finalize_node"]:
    """Decide if we should continue the loop or stop based on tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "finalize_node"

@observe(name="Finalize Node")
def finalize_node(state: MessagesState) -> MessagesState:
    final_message = AIMessage(content=state.get("messages", "")[-1].content if state.get("messages") else "No messages", tool_calls=[])
    return {
        "messages": [final_message],
        "llm_calls": state.get("llm_calls", 0),
        "used_tools": [],
    }


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)  # type: ignore
agent_builder.add_node("tool_node", tool_node)  # type: ignore
agent_builder.add_node("finalize_node", finalize_node)  # type: ignore

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "finalize_node"])
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("finalize_node", END)

agent = agent_builder.compile()


def run_pipeline(user_input: str) -> dict:
    final_state = agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "llm_calls": 0,
            "used_tools": [],
        }
    ) 
    langfuse.flush()
    return final_state


def extract_final_answer(final_state: dict) -> str:
    final_output = final_state.get("final_output")
    if isinstance(final_output, dict) and isinstance(final_output.get("answer"), str):
        return final_output["answer"]

    for message in reversed(final_state.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(part) for part in content)
            return str(content)
    return "Sem resposta final do agente."


def extract_structured_output(final_state: dict) -> AgentOutput:
    final_output = final_state.get("final_output")
    if isinstance(final_output, dict):
        return AgentOutput.model_validate(final_output)

    return AgentOutput(
        answer=extract_final_answer(final_state),
        success=True,
        used_tools=[],
    )

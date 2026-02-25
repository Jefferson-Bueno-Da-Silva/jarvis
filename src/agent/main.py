import json
import operator
import os

from typing import Literal
from langfuse import get_client, observe, Langfuse
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import Annotated, TypedDict
from pathlib import Path
from dotenv import load_dotenv

from src.models.model import GeminiModel, tools_by_name
from src.tools.tools import GOOGLE_TASKS_TOOLS
 
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
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

# @traceable(run_type="llm")
@observe(name="LLM Call")
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    # Create a nested generation for an LLM call
    messages = ""
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="llm-response",
        model="gemini-2.5-flash",
        input=SYS_PROMPT + state["messages"],
    ) as generation:
        # message = AIMessage(content="Mock llm response")
        messages = model_with_tools.invoke([SYS_PROMPT] + state["messages"])
        generation.update(output=messages, metadata=messages.response_metadata)

    return {
        "messages": [messages],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# @traceable(run_type="tool")
@observe(name="Tool Call")
def tool_node(state: dict):
    """"Performs the tool call decided by the LLM"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)
        if selected_tool is None:
            tool_output = {"ok": False, "error": f"Ferramenta não encontrada: {tool_name}"}
        else:
            tool_output = selected_tool.invoke(tool_call["args"])
        # print("Tool output:", tool_output)
        result.append(
            ToolMessage(
                content=json.dumps(tool_output, ensure_ascii=False),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )
    return {"messages": result}

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
# @traceable(run_type="retriever")
@observe(name="Should Continue?")
def should_continue(state: MessagesState) -> Literal["tool_node", END]: # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # print("Deciding whether to continue. Last message:", last_message)
    # If the LLM makes a tool call, then perform an action
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call) # type: ignore
agent_builder.add_node("tool_node", tool_node) # type: ignore

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# png_bytes = agent.get_graph(xray=True).draw_mermaid_png()
# salva na raiz do projeto (cwd)
# Path("graph_xray.png").write_bytes(png_bytes)

messages = [
    HumanMessage(
        # content="Liste as minhas tarefas",
        # content="faça um update a data da tarefa 'Comprar leite' para dia 27 de fevereiro de 2026 as 12:00"
        content="Apague a tarefa 'Comprar leite'"
        # content="Crie uma tarefa 'Comprar leite' para dia 26 de fevereiro de 2026 as 18:00"
    )
]

# @traceable
def run_pipeline():
    final_state = agent.invoke({"messages": messages}) # type: ignore
    langfuse.flush()
    return final_state

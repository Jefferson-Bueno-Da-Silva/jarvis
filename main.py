import json

from typing import Literal
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import Annotated, TypedDict
from pathlib import Path
import operator

from model import GeminiModel, tools_by_name
from tools import GOOGLE_TASKS_TOOLS

model = GeminiModel
model_with_tools = model.bind_tools(GOOGLE_TASKS_TOOLS)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

@traceable(run_type="llm")
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    print("LLM call with state:", state)
    return {
        "messages": [
                model_with_tools.invoke([
                    SystemMessage(
                        content=[
                            "You are a helpful assistant that can call tools to get information."
                            "translate the response to Portuguese.",
                            "Para obter o ID da tarefa, você pode primeiro usar a ferramenta para listar todas as tarefas e, em seguida, recuperar o ID da tarefa para executar o que foi solicitado."
                        ]
                    )
                ] 
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

@traceable(run_type="tool")
def tool_node(state: dict):
    """"Performs the tool call decided by the LLM"""
    print("Tool node with state:", state)
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)
        if selected_tool is None:
            tool_output = {"ok": False, "error": f"Ferramenta não encontrada: {tool_name}"}
        else:
            tool_output = selected_tool.invoke(tool_call["args"])

        print("Tool output:", tool_output)
        result.append(
            ToolMessage(
                content=json.dumps(tool_output, ensure_ascii=False),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )
    return {"messages": result}

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
@traceable(run_type="retriever")
def should_continue(state: MessagesState) -> Literal["tool_node", END]: # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    print("Deciding whether to continue. Last message:", last_message)

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
        content="faça um update a data da tarefa 'Comprar leite' para dia 27 de fevereiro de 2026 as 12:00"
    )
]

@traceable
def run_pipeline():
    final_state = agent.invoke({"messages": messages}) # type: ignore
    return final_state

final_state = run_pipeline()
for message in final_state["messages"]:
    message.pretty_print()

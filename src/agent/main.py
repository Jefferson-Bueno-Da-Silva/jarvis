from pathlib import Path

from langchain.messages import AIMessage, HumanMessage
from langfuse import propagate_attributes
from langgraph.graph import END, START, StateGraph

from src.agent.config import langfuse
from src.agent.nodes import bootstrap_tasks_node, finalize_node, llm_call, should_continue, tool_node
from src.agent.session import generate_session_id
from src.agent.state import AgentOutput, MessagesState


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("bootstrap_tasks_node", bootstrap_tasks_node)
agent_builder.add_node("llm_call", llm_call) 
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("finalize_node", finalize_node)

agent_builder.add_edge(START, "bootstrap_tasks_node")
agent_builder.add_edge("bootstrap_tasks_node", "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "finalize_node"])
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("finalize_node", END)

agent = agent_builder.compile()


def save_graph_image(output_path: str | Path = Path("src/static/graph_xray.png")) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    graph_png = agent.get_graph().draw_mermaid_png()
    target_path.write_bytes(graph_png)
    return target_path


def run_pipeline(user_input: str) -> dict:
    resolved_session_id = generate_session_id()
    with propagate_attributes(session_id=resolved_session_id):
        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "llm_calls": 0,
                "used_tools": [],
            }
        )
        langfuse.flush()
        return final_state

from pathlib import Path
import json
from typing import cast

from langchain.messages import AIMessage, HumanMessage
from langfuse import propagate_attributes
from langgraph.graph import END, START, StateGraph

from src.agent.config import langfuse
from src.agent.nodes import bootstrap_tasks_node, finalize_node, llm_call, should_continue, tool_node
from src.agent.session import generate_session_id
from src.agent.state import MessagesState


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


def _extract_text_from_state(state: MessagesState) -> str:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue

        content = message.content
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                        return parsed["text"]
                except json.JSONDecodeError:
                    pass
            return content

        if isinstance(content, list):
            # Gemini may return structured blocks like [{"type": "text", "text": "..."}]
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    return part["text"]
            return " ".join(str(part) for part in content)

        return str(content)
    return ""

def run_pipeline(user_input: str) -> dict:
    resolved_session_id = generate_session_id()
    with propagate_attributes(session_id=resolved_session_id):
        final_state = cast(
            MessagesState,
            agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "llm_calls": 0,
                "used_tools": [],
            }
            ),
        )
        langfuse.flush()
        return {
            "text": _extract_text_from_state(final_state),
            "llm_calls": final_state.get("llm_calls", 0),
            "used_tools": final_state.get("used_tools", []),
        }

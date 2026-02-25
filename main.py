from src.agent.main import run_pipeline

final_state = run_pipeline()
for message in final_state["messages"]:
    message.pretty_print()

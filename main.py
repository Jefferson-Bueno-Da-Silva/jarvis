import sys

from src.agent.main import run_pipeline


def main() -> None:
    user_input = " ".join(sys.argv[1:]).strip() or "Liste as minhas tarefas"
    final_state = run_pipeline(user_input)
    print(final_state)


if __name__ == "__main__":
    main()

.PHONY: help sync agent api

help:
	@echo "Targets disponíveis:"
	@echo "  make sync              # Instala/atualiza dependências"
	@echo "  make agent             # Executa o agente (mensagem padrão)"
	@echo "  make agent MSG='...'   # Executa o agente com mensagem customizada"
	@echo "  make api               # Sobe a API HTTP com reload"

sync:
	uv sync

agent:
	uv run python main.py "$(MSG)"

api:
	uv run uvicorn src.APP.main:app --reload

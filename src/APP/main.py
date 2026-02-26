from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.main import extract_final_answer, run_pipeline

app = FastAPI(
    title="Jarvis Agent API",
    version="1.0.0",
    description="HTTP API para interagir com o agente de Google Tasks.",
)


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Mensagem para o agente")


class AgentResponse(BaseModel):
    answer: str
    llm_calls: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
def ask_agent(payload: AgentRequest) -> AgentResponse:
    try:
        final_state = run_pipeline(payload.message)
        answer = extract_final_answer(final_state)
        return AgentResponse(answer=answer, llm_calls=final_state.get("llm_calls", 0))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {error}") from error

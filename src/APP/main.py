from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agent.main import extract_structured_output, run_pipeline

app = FastAPI(
    title="Jarvis Agent API",
    version="1.0.0",
    description="HTTP API para interagir com o agente de Google Tasks.",
)
GRAPH_IMAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "graph_xray.png"


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Mensagem para o agente")


class AgentResponse(BaseModel):
    answer: str
    success: bool
    used_tools: list[str]
    llm_calls: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph")
def graph_image() -> FileResponse:
    if not GRAPH_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Imagem do grafo não encontrada.")
    return FileResponse(path=GRAPH_IMAGE_PATH, media_type="image/png", filename="graph_xray.png")


@app.post("/agent", response_model=AgentResponse)
def ask_agent(payload: AgentRequest) -> AgentResponse:
    try:
        final_state = run_pipeline(payload.message)
        structured = extract_structured_output(final_state)
        return AgentResponse(
            answer=structured.answer,
            success=structured.success,
            used_tools=structured.used_tools,
            llm_calls=final_state.get("llm_calls", 0),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {error}") from error

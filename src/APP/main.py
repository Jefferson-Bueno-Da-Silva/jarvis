import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agent.main import run_pipeline

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
    llm_calls: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph")
def graph_image() -> FileResponse:
    if not GRAPH_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Imagem do grafo não encontrada.")
    return FileResponse(path=GRAPH_IMAGE_PATH, media_type="image/png", filename="graph_xray.png")


@app.post("/agent", response_model=dict)
def ask_agent(payload: AgentRequest) -> dict:
    try:
        final_state = run_pipeline(payload.message)
        return final_state
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {error}") from error


@app.post("/chat", response_model=dict)
async def chat_agent(request: Request) -> dict:
    data = dict(await request.form())
    print("Received form data:", data)
    message = data.get("Body", "")
    print("Extracted message:", message)
    return {"received": True}

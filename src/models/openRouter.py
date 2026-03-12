import os
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter

from src.tools.tools import GOOGLE_TASKS_TOOLS
from pydantic import SecretStr

load_dotenv()

_openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
OpenRouterModel = ChatOpenRouter(
    model="openrouter/free",
    temperature=0.8,
    api_key=SecretStr(_openrouter_api_key) if _openrouter_api_key is not None else None,
)
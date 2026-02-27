import os

from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langfuse import Langfuse

from src.models.model import GeminiModel
from src.tools.tools import GOOGLE_TASKS_TOOLS

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

model = GeminiModel
model_with_tools = model.bind_tools(GOOGLE_TASKS_TOOLS)

SYS_PROMPT = SystemMessage(
    content=[
        "You are a helpful assistant that can call tools to get information.",
        "translate the response to Portuguese.",
        "A snapshot from google_tasks_list may be preloaded in the conversation context before your first response.",
        "Prefer the preloaded snapshot before calling google_tasks_list again.",
        "To obtain the task ID, you can first use a tool to list all tasks and then retrieve the task ID to execute what was requested.",
    ]
)

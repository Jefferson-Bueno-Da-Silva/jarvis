import os

from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langfuse import Langfuse

from src.models.openRouter import OpenRouterModel
from src.tools.tools import GOOGLE_TASKS_TOOLS
from datetime import datetime

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

model = OpenRouterModel
model_with_tools = model.bind_tools(GOOGLE_TASKS_TOOLS)
today = datetime.now().astimezone()

SYS_PROMPT = SystemMessage(
    content="\n".join(
        [
            "You are a helpful assistant that can call tools to get information.",
            "translate the response to Portuguese.",
            "A snapshot from google_tasks_list may be preloaded in the conversation context before your first response.",
            "Prefer the preloaded snapshot before calling google_tasks_list again.",
            "To obtain the task ID, you can first use a tool to list all tasks and then retrieve the task ID to execute what was requested.",
            f"today is ${today}, every format of date and time should be DD/MM/YYYY and HH:MM, and the timezone should be the same as the one in the snapshot",
        ]
    )
)

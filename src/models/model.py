import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from src.tools.tools import GOOGLE_TASKS_TOOLS

load_dotenv()

GeminiModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

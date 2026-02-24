import json
import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import GOOGLE_TASKS_TOOLS

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
).bind_tools(GOOGLE_TASKS_TOOLS)

tools_by_name = {tool.name: tool for tool in GOOGLE_TASKS_TOOLS}

# Step 1: Model generates tool calls
messages: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [
    SystemMessage("You are a helpful assistant that can call tools to get information. translate the response to Portuguese"),
    HumanMessage("Liste as tarefas do Google Tasks para mim."),
]

ai_msg = model.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_name = tool_call["name"]
    selected_tool = tools_by_name.get(tool_name)
    if selected_tool is None:
        tool_output = {"ok": False, "error": f"Ferramenta não encontrada: {tool_name}"}
    else:
        tool_output = selected_tool.invoke(tool_call["args"])

    messages.append(
        ToolMessage(
            content=json.dumps(tool_output, ensure_ascii=False),
            tool_call_id=tool_call["id"],
            name=tool_name,
        )
    )

final_response = model.invoke(messages)
print(final_response.content)

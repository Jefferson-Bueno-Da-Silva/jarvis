import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Define the tool
@tool(description="Get the current weather in a given location")
def get_weather(location: str) -> str:
    return "It's sunny."

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
).bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [
    SystemMessage("You are a helpful assistant that can call tools to get information. translate the response to Portuguese"),
    HumanMessage("What's the weather in Boston?")
]

ai_msg = model.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model.invoke(messages)
print(final_response.content)
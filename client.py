from fastmcp import Client
from google import genai
import asyncio
from config import get_config_context

mcp_client = Client("server.py")
gemini_client = genai.Client()
model_name = "gemini-3.1-flash-lite-preview"



async def main(gemini_client):    
    async with mcp_client:
        response = await gemini_client.aio.models.generate_content(
            model=model_name,
            contents="Roll 3 dice!",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[mcp_client.session],  # Pass the FastMCP client session
            ),
        )
        print(response.text)

if __name__ == "__main__":
    


    config_context = get_config_context()
    env_vars = config_context.get("env", {})
    gemini_client = genai.Client(api_key= env_vars["GEMINI_API_KEY"])
    
    asyncio.run(main(gemini_client))
import asyncio
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    f = open('../keys/.openai.txt')
    OPENAI_API_KEY = SecretStr(f.read().strip())
    f.close()

    # client = MultiServerMCPClient(
    #     {
    #         "media": {
    #             "transport": "stdio",
    #             "command": "python",
    #             "args": ["mmapp_mcp_server.py"]
    #         }
    #     }
    # )

    client = MultiServerMCPClient(
        {
            "media": {
                "transport": "sse",
                "url": "http://127.0.0.1:8000/sse"
            }
        }
    )

    tools = await client.get_tools()

    SYSTEM_PROMPT = """You are a helpful AI media assistant.

    You have access to media tools.

    If the user asks to convert mp4 to mp3, extract audio, or create mp3 from a video,
    you MUST use the mp4_to_mp3 tool.

    Otherwise, answer normally.
    """

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                    model="gpt-4.1-mini", 
                    temperature=0.0,
                    max_completion_tokens=300)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    while True:
        user_input = input("\nMaigha: ").strip()
        if user_input.lower() == "exit":
            break

        response = await agent.ainvoke({
            "messages": [HumanMessage(content=user_input)]
        })

        for msg in response["messages"]:
            msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
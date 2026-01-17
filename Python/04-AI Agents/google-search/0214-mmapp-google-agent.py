from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())

f = open('../keys/.google.txt')
GOOGLE_API_KEY=f.read().strip()

f = open('../keys/.googlecse.txt')
GOOGLE_CSE_ID = f.read().strip()

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

SYSTEM_PROMPT = """You are a helpful AI assistant.
You are a helpful assistant.
Use google_search when the user asks for current, real-world, or factual information.
Otherwise, answer directly.
"""

google_search_tool  = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)


llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                model="gpt-4.1-mini", 
                temperature=0.0,
                max_completion_tokens=300)

agent = create_agent(
    model=llm,
    tools=[google_search_tool],
    system_prompt=SYSTEM_PROMPT
)

while True:
    user_input = input("\nMaigha: ").strip()
    if user_input.lower() == "exit":
        break

    response = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    for msg in response["messages"]:
        msg.pretty_print()
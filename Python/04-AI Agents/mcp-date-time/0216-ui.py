import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


# -------------------------
# Async init (cached)
# -------------------------
@st.cache_resource
def init_agent():

    f = open('../keys/.openai.txt')
    OPENAI_API_KEY = SecretStr(f.read().strip())
    f.close()

    client = MultiServerMCPClient(
        {
            "time": {
              "transport": "stdio",
              "command": "uvx",
              "args": [
                "mcp-server-time",
                "--local-timezone=America/New_York"
              ]
            }
        }
    )

    async def _init():
        tools = await client.get_tools()

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4.1-mini",
            temperature=0.0,
            max_completion_tokens=300
        )

        SYSTEM_PROMPT = """You are a helpful AI assistant.
You have access to tools.
Use tools whenever real-world or real-time information is required.
If a tool is not required, answer directly.
"""

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT
        )

        return agent

    return asyncio.run(_init())


agent = init_agent()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Maigha MCP Agent", page_icon="🤖", layout="wide")

st.title("🤖 Maigha MCP Agent (with Tools)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask something (time, date, etc.)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(
                agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })
            )

            ai_text = response["messages"][-1].content
            st.markdown(ai_text)

    st.session_state.messages.append({"role": "assistant", "content": ai_text})

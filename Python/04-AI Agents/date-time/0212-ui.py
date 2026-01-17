import streamlit as st
from datetime import datetime

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------
# Load API Key
# ----------------------------
f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())
f.close()

# ----------------------------
# Tool
# ----------------------------
@tool(name_or_callable="current_time", description="Give current date and time")
def current_datetime():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current date and time is {current_datetime}"

# ----------------------------
# System Prompt
# ----------------------------
SYSTEM_PROMPT = """You are a helpful AI assistant.
You have access to tools.
Use tools whenever real-world or real-time information is required.
If a tool is not required, answer directly.
"""

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

# ----------------------------
# Agent
# ----------------------------
agent = create_agent(
    model=llm,
    tools=[current_datetime],
    system_prompt=SYSTEM_PROMPT
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Maigha Agent", page_icon="🤖", layout="centered")

st.title("🤖 Maigha AI Agent")
st.caption("Tool-enabled agent (current time example)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent call
    response = agent.invoke({
        "messages": st.session_state.messages
    })

    # Only take the last AI message
    ai_msg = response["messages"][-1]
    st.session_state.messages.append(ai_msg)

    with st.chat_message("assistant"):
        st.markdown(ai_msg.content)

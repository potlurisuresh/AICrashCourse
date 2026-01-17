import streamlit as st

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

# ----------------------------
# Load Keys
# ----------------------------
f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())
f.close()

f = open('../keys/.google.txt')
GOOGLE_API_KEY = f.read().strip()
f.close()

f = open('../keys/.googlecse.txt')
GOOGLE_CSE_ID = f.read().strip()
f.close()

# ----------------------------
# Google Search Tool
# ----------------------------
search = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID
)

google_search_tool = Tool(
    name="google_search",
    description="Search Google for current or real-world information.",
    func=search.run,
)

# ----------------------------
# System Prompt
# ----------------------------
SYSTEM_PROMPT = """You are a helpful AI assistant.
Use google_search when the user asks for current, real-world, or factual information.
Otherwise, answer directly.
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
    tools=[google_search_tool],
    system_prompt=SYSTEM_PROMPT
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Maigha Web Agent", page_icon="🌐", layout="centered")
st.title("🌐 Maigha Web-Augmented Agent")
st.caption("Agent + Google Search Tool")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask about current events, news, facts...")

if user_input:
    # Save user msg
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert to LangChain messages
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Agent call
    response = agent.invoke({"messages": lc_messages})
    ai_text = response["messages"][-1].content

    # Save assistant msg
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_text
    })

    with st.chat_message("assistant"):
        st.markdown(ai_text)

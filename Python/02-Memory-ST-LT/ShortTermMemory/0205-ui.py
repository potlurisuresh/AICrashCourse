import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain_core.runnables.history import RunnableWithMessageHistory

# ----------------------------
# Load API key
# ----------------------------
with open("../keys/.openai.txt") as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# ----------------------------
# Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Maigha Assistant"),
    ("human", "{topic}")
])

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

chain = prompt | llm | StrOutputParser()

# ----------------------------
# In-memory session store
# ----------------------------
if "store" not in st.session_state:
    st.session_state.store = {}

# ----------------------------
# DB setup
# ----------------------------
db_path = "chat_data/sql_chat_history.db"
engine = create_engine(f"sqlite:///{db_path}")

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id,connection=engine)

chain_with_db_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="topic"
)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Maigha Chat", page_icon="🤖", layout="centered")
st.title("Maigha")

# ---- Sidebar: Session control ----
st.sidebar.title("💬 Sessions")

if "session_id" not in st.session_state:
    st.session_state.session_id = "maigha-session"

new_session = st.sidebar.text_input("Create Session", st.session_state.session_id)

if st.sidebar.button("Load Session"):
    st.session_state.session_id = new_session

st.sidebar.markdown("### Existing Sessions")
for s in st.session_state.store.keys():
    if st.sidebar.button(s):
        st.session_state.session_id = s

st.sidebar.markdown(f"**Active Session:** `{st.session_state.session_id}`")

# ----------------------------
# Show chat history
# ----------------------------
history = get_session_history(st.session_state.session_id)

for msg in history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask Maigha...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    output = chain_with_db_memory.invoke(
        {"topic": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    with st.chat_message("assistant"):
        st.markdown(output)


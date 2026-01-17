import os
import streamlit as st
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma


# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Maigha AI", page_icon="🧠", layout="wide")
st.title("🧠 Maigha – Memory Augmented AI")


# ---------------------------------
# Ensure folders exist
# ---------------------------------
os.makedirs("chat_data", exist_ok=True)
os.makedirs("chat_data/long_term_memory", exist_ok=True)


# ---------------------------------
# Load API key
# ---------------------------------
with open("../keys/.openai.txt") as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())


# ---------------------------------
# Prompt
# ---------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical assistant"),
    ("system", "Relevant long term memory:\n{long_memory}"),
    ("human", "{topic}")
])


# ---------------------------------
# LLM + Chain
# ---------------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

chain = prompt | llm | StrOutputParser()


# ---------------------------------
# Short-term memory (SQLite)
# ---------------------------------
db_path = "chat_data/sql_chat_history.db"
engine = create_engine(f"sqlite:///{db_path}")

def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine
    )

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="topic"
)


# ---------------------------------
# Long-term memory (Chroma)
# ---------------------------------
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

vectorstore = Chroma(
    collection_name="maigha_memory",
    persist_directory="chat_data/long_term_memory",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def save_long_term(text, session_id):
    vectorstore.add_texts([f"[{session_id}] {text}"])

def recall_long_term(query):
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)


# ---------------------------------
# Session utilities
# ---------------------------------
def get_db_sessions():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT DISTINCT session_id FROM message_store")
            )
            return [row[0] for row in result.fetchall()]
    except Exception:
        return []


# ---------------------------------
# Init local session registry
# ---------------------------------
if "known_sessions" not in st.session_state:
    st.session_state.known_sessions = ["maigha-session"]

if "session_id" not in st.session_state:
    st.session_state.session_id = "maigha-session"


# ---------------------------------
# Sidebar – Session Manager
# ---------------------------------
st.sidebar.header("🗂 Chat Sessions")

db_sessions = get_db_sessions()

# Merge DB + local
all_sessions = sorted(set(st.session_state.known_sessions + db_sessions))

# New session
new_session = st.sidebar.text_input("Create new session")

if st.sidebar.button("➕ Create / Switch"):
    if new_session.strip():
        name = new_session.strip()
        if name not in st.session_state.known_sessions:
            st.session_state.known_sessions.append(name)
        st.session_state.session_id = name
        st.rerun()

# Select existing
selected = st.sidebar.selectbox(
    "Select existing session",
    options=all_sessions,
    index=all_sessions.index(st.session_state.session_id)
    if st.session_state.session_id in all_sessions else 0
)

if selected != st.session_state.session_id:
    st.session_state.session_id = selected
    st.rerun()

st.sidebar.markdown(f"**Active session:** `{st.session_state.session_id}`")


# ---------------------------------
# Load conversation history
# ---------------------------------
history = get_session_history(st.session_state.session_id)

for msg in history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# ---------------------------------
# Chat input
# ---------------------------------
user_input = st.chat_input("Ask Maigha...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    long_memory = recall_long_term(user_input)

    response = chain_with_memory.invoke(
        {"topic": user_input, "long_memory": long_memory},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    save_long_term(user_input, st.session_state.session_id)
    save_long_term(response, st.session_state.session_id)

    # persist session
    if st.session_state.session_id not in st.session_state.known_sessions:
        st.session_state.known_sessions.append(st.session_state.session_id)

    with st.chat_message("assistant"):
        st.markdown(response)

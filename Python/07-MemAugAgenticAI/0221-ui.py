import os
import subprocess
from typing import TypedDict, Optional
from datetime import datetime

import streamlit as st

from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from sqlalchemy import create_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"
TESTPLAN_PATH = "MultiMagixTestPlan.txt"
PERSIST_DIR = "rag_data/testplan"

LONG_MEMORY_DIR = "chat_data/long_term_memory"
DB_PATH = "chat_data/sql_chat_history.db"
DEFAULT_SESSION = "maigha-session"


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Maigha AI", page_icon="🧠", layout="wide")
st.title("🧠 Maigha AI — Memory Enabled Multi-Agent System")
st.caption("Agentic RAG + Tool Execution + Short & Long Term Memory")


# ============================================================
# API KEY
# ============================================================

OPENAI_API_KEY = SecretStr(open('../keys/.openai.txt').read().strip())


# ============================================================
# MODELS
# ============================================================

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0.0)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")


# ============================================================
# MEMORY
# ============================================================

engine = create_engine(f"sqlite:///{DB_PATH}")

def get_session_history(session_id: str):
    """Return short-term SQL-backed chat history for a session."""
    return SQLChatMessageHistory(session_id=session_id, connection=engine)


long_memory_vs = Chroma(
    collection_name="maigha_long_memory",
    persist_directory=LONG_MEMORY_DIR,
    embedding_function=embeddings
)

long_memory_retriever = long_memory_vs.as_retriever(search_kwargs={"k": 3})


def recall_long_term(query: str) -> str:
    """Retrieve relevant long-term memories from vector store."""
    docs = long_memory_retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)


def save_long_term(text: str, session_id: str):
    """Store text into long-term vector memory under a session namespace."""
    long_memory_vs.add_texts([f"[{session_id}] {text}"])


# ============================================================
# SESSION HELPERS (PERSISTENCE)
# ============================================================

def load_chat_from_db(session_id: str):
    """Load full chat history for a session from SQL memory."""
    history = get_session_history(session_id)
    messages = []

    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        messages.append((role, msg.content))

    return messages


def list_sessions_from_db():
    """Return all distinct session IDs from the SQL memory store."""
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT session_id FROM message_store")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


# ============================================================
# BUILD / LOAD TEST PLAN RAG
# ============================================================

def load_testplan_retriever():
    """Load or build the Chroma retriever for the MMApp test plan."""
    vectorstore = Chroma(
        collection_name="maigha_testplan_memory",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        docs = TextLoader(TESTPLAN_PATH, encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(docs)
        vectorstore.add_documents(chunks)

    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = load_testplan_retriever()


# ============================================================
# TOOLS
# ============================================================

@tool
def search_testplan(query: str) -> str:
    """Search the MultiMagix test plan using vector similarity."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# AGENTS
# ============================================================

intent_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="Classify intent: testplan | time | general. Return one word."
)

testplan_agent = create_agent(
    model=llm,
    tools=[search_testplan],
    system_prompt="""
You are the TestPlan Agent.

If conversion → output ONLY executable MMApp.exe command.
Else → explain from test plan.
If unsupported → NOT_SUPPORTED.
"""
)

general_agent = create_agent(llm, [], system_prompt="You are a helpful general AI assistant.")
time_agent = create_agent(llm, [], system_prompt="When asked, tell the current date and time.")


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    user_input: str
    long_memory: str
    intent: Optional[str]
    testplan_output: Optional[str]
    final_output: Optional[str]


# ============================================================
# NODES
# ============================================================

def intent_node(state: AgentState) -> AgentState:
    res = intent_agent.invoke({
        "messages": [HumanMessage(content=f"Memory:\n{state['long_memory']}\n\nUser: {state['user_input']}")]
    })
    return {**state, "intent": res["messages"][-1].content.strip().lower()}


def testplan_node(state: AgentState) -> AgentState:
    res = testplan_agent.invoke({
        "messages": [HumanMessage(content=f"Memory:\n{state['long_memory']}\n\nUser: {state['user_input']}")]
    })
    return {**state, "testplan_output": res["messages"][-1].content.strip()}


def is_safe_mmapp_command(cmd: str) -> bool:
    return cmd.startswith("MMApp.exe") and all(x not in cmd for x in ["&", "|", "&&", ";"])


def executor_node(state: AgentState) -> AgentState:
    command = state.get("testplan_output")

    if not command:
        return {**state, "final_output": "❌ No command to execute."}

    command = command.strip()

    if not is_safe_mmapp_command(command):
        return {**state, "final_output": "❌ Blocked unsafe command."}

    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    result = subprocess.run(command, capture_output=True, text=True, shell=False)

    if result.returncode != 0:
        return {**state, "final_output": result.stderr}

    mp3s = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]
    return {**state, "final_output": f"SUCCESS :: {os.path.join(OUTPUT_DIR, mp3s[0])}" if mp3s else "No MP3 generated."}
            
            
def time_node(state: AgentState) -> AgentState:
        return {**state, "final_output": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def general_node(state: AgentState) -> AgentState:
    res = general_agent.invoke({
        "messages": [HumanMessage(content=f"Memory:\n{state['long_memory']}\n\nUser: {state['user_input']}")]
    })
    return {**state, "final_output": res["messages"][-1].content}


# ============================================================
# ROUTERS
# ============================================================

def route_from_intent(state: AgentState):
    return state["intent"]


def route_from_testplan(state: AgentState):
    out = state.get("testplan_output")

    if not out:
        return "end"

    out = out.strip()

    if out == "NOT_SUPPORTED":
        return "end"

    if out.startswith("MMApp.exe"):
        return "execute"

    return "end"


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("intent", intent_node)
graph.add_node("testplan", testplan_node)
graph.add_node("executor", executor_node)
graph.add_node("time", time_node)
graph.add_node("general", general_node)

graph.set_entry_point("intent")

graph.add_conditional_edges("intent", route_from_intent, {
    "testplan": "testplan",
    "time": "time",
    "general": "general"
})

graph.add_conditional_edges("testplan", route_from_testplan, {
    "execute": "executor",
    "end": END
})

graph.add_edge("executor", END)
graph.add_edge("time", END)
graph.add_edge("general", END)

app = graph.compile()


# ============================================================
# SESSION UI (DROPDOWN + PERSISTENCE)
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = DEFAULT_SESSION

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

existing_sessions = list_sessions_from_db()
if DEFAULT_SESSION not in existing_sessions:
    existing_sessions.insert(0, DEFAULT_SESSION)

with st.sidebar:
    st.header("🗂️ Sessions")

    selected = st.selectbox(
        "Select session",
        existing_sessions,
        index=existing_sessions.index(st.session_state.session_id)
        if st.session_state.session_id in existing_sessions else 0
    )

    if selected != st.session_state.session_id:
        st.session_state.session_id = selected
        st.session_state.chat_log = load_chat_from_db(selected) # type: ignore
        st.rerun()

    st.markdown("### ➕ Create new session")
    new_session = st.text_input("Session name")

    if st.button("Start new session") and new_session.strip():
        st.session_state.session_id = new_session.strip()
        st.session_state.chat_log = []
        st.rerun()

    st.markdown("---")
    st.markdown(f"**Active session:** `{st.session_state.session_id}`")

if st.session_state.chat_log == []:
    st.session_state.chat_log = load_chat_from_db(st.session_state.session_id)


# ============================================================
# STREAMLIT CHAT UI
# ============================================================

for role, msg in st.session_state.chat_log:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("Ask Maigha...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Maigha is thinking..."):

        long_memory = recall_long_term(user_input)

        result = app.invoke({
            "user_input": user_input,
            "long_memory": long_memory,
            "intent": None,
            "testplan_output": None,
            "final_output": None
        })

        save_long_term(user_input, st.session_state.session_id)
        save_long_term(result["final_output"], st.session_state.session_id)

        history = get_session_history(st.session_state.session_id)
        history.add_user_message(user_input)
        history.add_ai_message(result["final_output"])

        st.session_state.chat_log.append(("user", user_input))
        st.session_state.chat_log.append(("assistant", result["final_output"]))

        with st.chat_message("assistant"):
            st.markdown(result["final_output"])

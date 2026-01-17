import streamlit as st
import os
import subprocess
from typing import TypedDict, Optional
from datetime import datetime

from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END


# ============================================================
# CONFIG
# ============================================================
MMAPP_PATH = r"C:\\MMWS\\MMMediaSuite\\MMApp\\Console\\Windows\\x64\Debug\\"
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"
TESTPLAN_PATH = "MultiMagixTestPlan.txt"
PERSIST_DIR = "rag_data/testplan"


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Maigha AI", page_icon="🎬", layout="wide")
st.title("🎬 Maigha Multi-Agent AI")
st.caption("Intent Agent • TestPlan RAG Agent • Executor • Time Agent")


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
# BUILD / LOAD TEST PLAN RAG
# ============================================================
@st.cache_resource
def load_testplan_retriever():
    vectorstore = Chroma(
        collection_name="maigha_testplan_memory",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        st.info("📌 Indexing test plan (first run)...")

        docs = TextLoader(TESTPLAN_PATH, encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=30
        ).split_documents(docs)

        vectorstore.add_documents(chunks)

    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = load_testplan_retriever()


# ============================================================
# TOOLS
# ============================================================
@tool
def search_testplan(query: str) -> str:
    """
    Search the MultiMagix test plan for workflows, commands,
    validation steps and MMApp usage.
    """
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# AGENTS
# ============================================================
intent_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""
Classify user request into exactly one word:
testplan, time, general
"""
)

testplan_agent = create_agent(
    model=llm,
    tools=[search_testplan],
    system_prompt="""
You are the TestPlan Agent.

If user asks about workflows or usage → answer.

If user asks to convert media:
→ Output ONLY the executable MMApp command.
→ No explanations. No markdown.
"""
)

general_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful general assistant."
)

time_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="Tell current date and time."
)


# ============================================================
# STATE
# ============================================================
class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    testplan_output: Optional[str]
    final_output: Optional[str]


# ============================================================
# LANGGRAPH NODES
# ============================================================
def intent_node(state: AgentState) -> AgentState:
    res = intent_agent.invoke({"messages":[HumanMessage(content=state["user_input"])]})
    return {**state, "intent": res["messages"][-1].content.strip().lower()}


def testplan_node(state: AgentState) -> AgentState:
    res = testplan_agent.invoke({"messages":[HumanMessage(content=state["user_input"])]})
    return {**state, "testplan_output": res["messages"][-1].content.strip()}


def executor_node(state: AgentState) -> AgentState:
    cmd = state["testplan_output"]
    cmd = MMAPP_PATH + cmd # type: ignore
    print(cmd)
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    result = subprocess.run(cmd, capture_output=True, text=True, shell=False) # type: ignore

    if result.returncode != 0:
        return {**state, "final_output": result.stderr}

    mp3s = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]
    if not mp3s:
        return {**state, "final_output": "No MP3 generated."}

    return {**state, "final_output": f"SUCCESS :: {os.path.join(OUTPUT_DIR, mp3s[0])}"}


def time_node(state: AgentState) -> AgentState:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {**state, "final_output": now}


def general_node(state: AgentState) -> AgentState:
    res = general_agent.invoke({"messages":[HumanMessage(content=state["user_input"])]})
    return {**state, "final_output": res["messages"][-1].content}


# ============================================================
# ROUTERS
# ============================================================
def route_from_intent(state: AgentState) -> str:
    return state["intent"] # type: ignore


def route_from_testplan(state: AgentState) -> str:
    if state["testplan_output"].lower().startswith("mmapp"): # type: ignore
        return "execute"
    return "end"


# ============================================================
# LANGGRAPH BUILD
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.subheader("🧩 LangGraph")
    st.code(app.get_graph().draw_ascii())


# ============================================================
# CHAT UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about test plan, conversion, or time...")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    result = app.invoke({
        "user_input": user_input,
        "intent": None,
        "testplan_output": None,
        "final_output": None
    })

    output = result["final_output"]

    st.session_state.messages.append({"role":"assistant","content":output})

    with st.chat_message("assistant"):
        st.markdown(output)

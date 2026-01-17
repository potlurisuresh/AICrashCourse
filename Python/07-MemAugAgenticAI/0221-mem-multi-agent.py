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

# ===== MEMORY IMPORTS (ADDED) =====
from sqlalchemy import create_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory


# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"
TESTPLAN_PATH = "MultiMagixTestPlan.txt"
PERSIST_DIR = "rag_data/testplan"

# ===== MEMORY CONFIG (ADDED) =====
LONG_MEMORY_DIR = "chat_data/long_term_memory"
DB_PATH = "chat_data/sql_chat_history.db"
SESSION_ID = "maigha-session"


# ============================================================
# API KEY
# ============================================================
OPENAI_API_KEY = SecretStr(open('../keys/.openai.txt').read().strip())


# ============================================================
# MODELS
# ============================================================
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0
)

embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)


# ============================================================
# ================= MEMORY (ADDED) ==========================
# ============================================================

# ---------- Short-term memory (SQL) ----------
engine = create_engine(f"sqlite:///{DB_PATH}")

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection=engine)


# ---------- Long-term memory (Vector DB) ----------
long_memory_vs = Chroma(
    collection_name="maigha_long_memory",
    persist_directory=LONG_MEMORY_DIR,
    embedding_function=embeddings
)

long_memory_retriever = long_memory_vs.as_retriever(search_kwargs={"k": 3})

def recall_long_term(query: str) -> str:
    docs = long_memory_retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)

def save_long_term(text: str, session_id: str):
    long_memory_vs.add_texts([f"[{session_id}] {text}"])


# ============================================================
# BUILD / LOAD TEST PLAN RAG
# ============================================================
def load_testplan_retriever():
    vectorstore = Chroma(
        collection_name="maigha_testplan_memory",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        print("📌 Indexing test plan...")

        docs = TextLoader(TESTPLAN_PATH, encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
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
    Search the MultiMagix test plan.

    Use when the user asks about:
    - workflows
    - commands
    - validation steps
    - MMApp usage
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
You are the Intent Agent.

Classify the user request into exactly one word:

- testplan
- time
- general

Return only one word.
"""
)

testplan_agent = create_agent(
    model=llm,
    tools=[search_testplan],
    system_prompt="""
You are the TestPlan Agent for MMApp.

You MUST follow these rules strictly.

============================
MODE SELECTION (MANDATORY)
============================

MODE A — KNOWLEDGE MODE
Use if the user asks about:
- workflows
- validation steps
- how MMApp works

Output:
- Normal explanation is allowed.

----------------------------

MODE B — COMMAND MODE
Use ONLY if the user intent is to perform a media operation
(example: convert, extract, compress, merge, split, mp4 to mp3, etc.)

In MODE B you MUST:

1. Output ONLY ONE SINGLE LINE.
2. That line MUST start with: MMApp.exe
3. That line MUST be a fully executable command.
4. NO markdown.
5. NO explanation.
6. NO quotes.
7. NO emojis.
8. NO prefixes like "Command:".

Correct:
MMApp.exe -f 1 -i C:\\input.mp4 -o 1

Wrong:
Here is the command: MMApp.exe ...
```MMApp.exe ...```
MMApp.exe ... (this converts mp4 to mp3)

----------------------------

============================
SOURCE OF TRUTH
============================

- You may ONLY use the test plan (via search_testplan).
- NEVER invent flags or workflows.
- If a command cannot be constructed, output exactly:

NOT_SUPPORTED

----------------------------

FINAL SELF CHECK (MANDATORY)
- Does output start with MMApp.exe OR is it NOT_SUPPORTED?
- Is it one single line?
- No commentary?

Fix internally before answering.
"""
)

general_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful general AI assistant."
)

time_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="When asked, tell the current date and time."
)


# ============================================================
# STATE (MINIMAL CHANGE)
# ============================================================
class AgentState(TypedDict):
    user_input: str
    long_memory: str
    intent: Optional[str]
    testplan_output: Optional[str]
    final_output: Optional[str]


# ============================================================
# NODES (MINIMALLY MODIFIED)
# ============================================================
def intent_node(state: AgentState) -> AgentState:
    res = intent_agent.invoke({
        "messages": [HumanMessage(content=f"""
Relevant memory:
{state['long_memory']}

User: {state['user_input']}
""")]
    })
    return {**state, "intent": res["messages"][-1].content.strip().lower()}


def testplan_node(state: AgentState) -> AgentState:
    res = testplan_agent.invoke({
        "messages": [HumanMessage(content=f"""
Relevant memory:
{state['long_memory']}

User: {state['user_input']}
""")]
    })
    return {**state, "testplan_output": res["messages"][-1].content.strip()}


def is_safe_mmapp_command(cmd: str) -> bool:
    return (
        cmd.startswith("MMApp.exe")
        and "&" not in cmd
        and "|" not in cmd
        and "&&" not in cmd
        and ";" not in cmd
    )


def executor_node(state: AgentState) -> AgentState:
    command = state["testplan_output"].strip()  # type: ignore

    if not is_safe_mmapp_command(command):
        return {**state, "final_output": "❌ Blocked: Invalid or unsafe MMApp command."}

    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    result = subprocess.run(command, capture_output=True, text=True, shell=False)

    if result.returncode != 0:
        return {**state, "final_output": f"❌ MMApp failed:\n{result.stderr}"}

    mp3s = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]

    if not mp3s:
        return {**state, "final_output": "❌ Execution finished but no MP3 found."}

    return {**state, "final_output": f"SUCCESS :: {os.path.join(OUTPUT_DIR, mp3s[0])}"}


def time_node(state: AgentState) -> AgentState:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {**state, "final_output": f"🕒 Current date and time: {now}"}


def general_node(state: AgentState) -> AgentState:
    res = general_agent.invoke({
        "messages": [HumanMessage(content=f"""
Conversation memory:
{state['long_memory']}

User: {state['user_input']}
""")]
    })
    return {**state, "final_output": res["messages"][-1].content}


# ============================================================
# ROUTERS (UNCHANGED)
# ============================================================
def route_from_intent(state: AgentState) -> str:
    return state["intent"]  # type: ignore


def route_from_testplan(state: AgentState) -> str:
    out = state["testplan_output"].strip()  # type: ignore

    if out == "NOT_SUPPORTED":
        return "end"

    if out.startswith("MMApp.exe"):
        return "execute"

    return "end"


# ============================================================
# GRAPH (UNCHANGED)
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

print("\n🧩 LANGGRAPH STRUCTURE:\n")
print(app.get_graph().draw_mermaid())


# ============================================================
# MAIN LOOP (MEMORY HOOKED)
# ============================================================
print("\n🧠 Maigha AI Ready (Memory Enabled)\n")

while True:
    q = input("\nMaigha > ").strip()
    if q.lower() == "exit":
        break

    long_memory = recall_long_term(q)

    result = app.invoke({
        "user_input": q,    
        "long_memory": long_memory,
        "intent": None,
        "testplan_output": None,
        "final_output": None
    })

    # ---- persist memory ----
    save_long_term(q, SESSION_ID)
    save_long_term(result["final_output"], SESSION_ID)

    history = get_session_history(SESSION_ID)
    history.add_user_message(q)
    history.add_ai_message(result["final_output"])

    print("\n✅ Output:\n", result["final_output"])

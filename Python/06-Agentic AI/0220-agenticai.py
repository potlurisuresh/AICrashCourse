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
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"
TESTPLAN_PATH = "MultiMagixTestPlan.txt"
PERSIST_DIR = "rag_data/testplan"


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

    Use this tool when the user asks about:
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
You are the TestPlan Agent.

----------------------------
MODE 1: Knowledge
----------------------------
If the user asks about workflows, validation, or usage:
→ Answer using the test plan.

----------------------------
MODE 2: Execution command
----------------------------
If the user asks to convert media (mp4 to mp3):

→ Output ONLY the final executable command.
→ NO explanations.
→ NO markdown.
→ NO prefixes.

Correct:
MMApp.exe -f 1 -i C:\\input.mp4 -o 1

Never invent commands.
Never execute commands.
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
    res = intent_agent.invoke({
        "messages": [HumanMessage(content=state["user_input"])]
    })
    return {**state, "intent": res["messages"][-1].content.strip().lower()}


def testplan_node(state: AgentState) -> AgentState:
    res = testplan_agent.invoke({
        "messages": [HumanMessage(content=state["user_input"])]
    })
    return {**state, "testplan_output": res["messages"][-1].content.strip()}


def executor_node(state: AgentState) -> AgentState:
    command = state["testplan_output"]

    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    result = subprocess.run(command, capture_output=True, text=True, shell=False) # type: ignore

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
        "messages": [HumanMessage(content=state["user_input"])]
    })
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

print("\n🧩 LANGGRAPH STRUCTURE:\n")
print(app.get_graph().draw_mermaid())


# ============================================================
# MAIN LOOP
# ============================================================
print("\n🧠 Maigha AI Ready (STRICT command mode)\n")

while True:
    q = input("\nMaigha > ").strip()
    if q.lower() == "exit":
        break

    result = app.invoke({
        "user_input": q,
        "intent": None,
        "testplan_output": None,
        "final_output": None
    })

    print("\n✅ Output:\n", result["final_output"])

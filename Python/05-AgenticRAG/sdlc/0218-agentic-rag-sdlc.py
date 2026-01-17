import os
from pydantic import SecretStr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


# ============================
# Software Dev Assistant (Agentic RAG)
# ============================

# ----------------------------
# API key
# ----------------------------
with open('../keys/.openai.txt') as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# ----------------------------
# Files & DB paths
# ----------------------------
TESTPLAN_FILE = "MultiMagixTestPlan.txt"
CODE_FILE = "MMMSSpeed.cpp"

TESTPLAN_DIR = "data/testplan"
CODE_DIR = "data/code"

os.makedirs(TESTPLAN_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

# ----------------------------
# 1. Load documents
# ----------------------------
print("\n🔹 Loading documents...")

testplan_docs = TextLoader(TESTPLAN_FILE, encoding="utf-8").load()
code_docs = TextLoader(CODE_FILE, encoding="utf-8").load()

# ----------------------------
# 2. Split into chunks
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

testplan_chunks = splitter.split_documents(testplan_docs)
code_chunks = splitter.split_documents(code_docs)

# ----------------------------
# 3. Embeddings
# ----------------------------
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# ----------------------------
# 4. Vector stores
# ----------------------------
testplan_vs = Chroma(
    collection_name="software_testplan_memory",
    persist_directory=TESTPLAN_DIR,
    embedding_function=embeddings
)

code_vs = Chroma(
    collection_name="software_code_memory",
    persist_directory=CODE_DIR,
    embedding_function=embeddings
)

# ----------------------------
# 5. Add documents only once
# ----------------------------
if testplan_vs._collection.count() == 0:
    print("📘 Indexing test plan...")
    testplan_vs.add_documents(testplan_chunks)

if code_vs._collection.count() == 0:
    print("💻 Indexing source code...")
    code_vs.add_documents(code_chunks)

# ----------------------------
# 6. Retrievers
# ----------------------------
testplan_retriever = testplan_vs.as_retriever(search_kwargs={"k": 5})
code_retriever = code_vs.as_retriever(search_kwargs={"k": 5})

# ----------------------------
# 7. Prompts + Chains
# ----------------------------
testplan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a software test engineer assistant. Answer ONLY from the test plan."),
    ("system", "Test plan context:\n{context}"),
    ("user", "{question}")
])

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a C++ software engineer assistant. Answer ONLY from the given source code."),
    ("system", "Source code context:\n{context}"),
    ("user", "{question}")
])

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0
)

testplan_chain = testplan_prompt | llm | StrOutputParser()
code_chain = code_prompt | llm | StrOutputParser()

# ----------------------------
# 8. Tools (each RAG = tool)
# ----------------------------
@tool
def ask_testplan(question: str) -> str:
    """Use this for questions about test cases, commands, validation, workflows, and usage."""
    docs = testplan_retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    return testplan_chain.invoke({
        "question": question,
        "context": context
    })


@tool
def ask_codebase(question: str) -> str:
    """Use this for questions about C++ code, functions, logic, bugs, and implementation."""
    docs = code_retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    return code_chain.invoke({
        "question": question,
        "context": context
    })

# ----------------------------
# 9. Router Agent
# ----------------------------
SYSTEM_PROMPT = """
You are a Software Development Assistant.

You have two knowledge sources:
1. Test Plan (testing, validation, commands, workflows, how to use the software)
2. Source Code (C++ implementation, functions, logic, bugs)

Routing rules:
- If the question is about tests, validation, commands, workflows → use ask_testplan
- If the question is about code, logic, functions, internals → use ask_codebase
- Always use one appropriate tool to answer.
"""

agent = create_agent(
    model=llm,
    tools=[ask_testplan, ask_codebase],
    system_prompt=SYSTEM_PROMPT
)

# ----------------------------
# 10. CLI loop
# ----------------------------
print("\n🧠 Software Dev Assistant (Agentic RAG)")
print("Ask anything about test plan or code. Type 'exit' to quit.")

while True:
    q = input("\nAsk: ").strip()
    if q.lower() == "exit":
        break

    response = agent.invoke({
        "messages": [HumanMessage(content=q)]
    })

    print("\n✅ Answer:\n", response["messages"][-1].content)

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr

# ============================
# Software Dev Assistant RAG
# ============================

# ----------------------------
# API key
# ----------------------------
with open('../keys/.openai.txt') as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# ----------------------------
# Files
# ----------------------------
TESTPLAN_FILE = "MultiMagixTestPlan.txt"
CODE_FILE = "MMMSSpeed.cpp"
TESTPLAN_DIR = "data/testplan"
CODE_DIR = "data/code"

# ----------------------------
# 1. Load documents
# ----------------------------
print("\n🔹 Loading software documents...")

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

# print("Test plan chunks:", len(testplan_chunks))
# print("Code chunks     :", len(code_chunks))

# ----------------------------
# 3. Embeddings
# ----------------------------
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# ----------------------------
# 4. Create vector stores (separate brains)
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
# 7. Prompts
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
# 8. CLI loop
# ----------------------------
while True:

    print("\n==============================")
    print("🧠 Software Dev Assistant RAG")
    print("==============================")
    print("1. Ask Test Plan")
    print("2. Ask Source Code")
    print("exit")

    mode = input("\nSelect mode: ").strip()

    if mode.lower() == "exit":
        break

    if mode not in ["1", "2"]:
        print("❌ Invalid option")
        continue

    if mode == "1":
        retriever = testplan_retriever
        chain = testplan_chain
        print("\n📘 Test Plan Assistant ready")

    else:
        retriever = code_retriever
        chain = code_chain
        print("\n💻 Code Assistant ready")

    while True:
        q = input("\nAsk (or 'back'): ").strip()
        if q.lower() == "back":
            break

        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content for d in docs)

        answer = chain.invoke({
            "question": q,
            "context": context
        })

        print("\n✅ Answer:\n", answer)

import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Software Dev Assistant RAG", page_icon="🧠", layout="wide")
st.title("🧠 Software Dev Assistant RAG")
st.caption("Test Plans + Source Code + Retrieval-Augmented Generation")

# ---------------------------------
# API key
# ---------------------------------
with open('../keys/.openai.txt') as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# ---------------------------------
# Files & folders
# ---------------------------------
TESTPLAN_FILE = "MultiMagixTestPlan.txt"
CODE_FILE = "MMMSSpeed.cpp"
TESTPLAN_DIR = "data/testplan"
CODE_DIR = "data/code"

# ---------------------------------
# Core components
# ---------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0
)

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

testplan_chain = testplan_prompt | llm | StrOutputParser()
code_chain = code_prompt | llm | StrOutputParser()

# ---------------------------------
# Build / load vector stores
# ---------------------------------
@st.cache_resource(show_spinner=True)
def prepare_vectorstores():

    # Load docs
    testplan_docs = TextLoader(TESTPLAN_FILE, encoding="utf-8").load()
    code_docs = TextLoader(CODE_FILE, encoding="utf-8").load()

    # Split
    testplan_chunks = splitter.split_documents(testplan_docs)
    code_chunks = splitter.split_documents(code_docs)

    # Vector DBs
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

    # Index once
    if testplan_vs._collection.count() == 0:
        testplan_vs.add_documents(testplan_chunks)

    if code_vs._collection.count() == 0:
        code_vs.add_documents(code_chunks)

    return testplan_vs, code_vs

with st.spinner("Loading software knowledge bases..."):
    testplan_vs, code_vs = prepare_vectorstores()

# ---------------------------------
# UI – Knowledge selector
# ---------------------------------
mode = st.selectbox("🗂️ Select knowledge base", ["Test Plan", "Source Code"])

if mode == "Test Plan":
    retriever = testplan_vs.as_retriever(search_kwargs={"k": 5})
    chain = testplan_chain
    st.success("📘 Test Plan Assistant active")
else:
    retriever = code_vs.as_retriever(search_kwargs={"k": 5})
    chain = code_chain
    st.success("💻 Source Code Assistant active")

st.divider()

# ---------------------------------
# Chat memory
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------
# Chat input
# ---------------------------------
question = st.chat_input("Ask something...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        answer = chain.invoke({
            "question": question,
            "context": context
        })

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

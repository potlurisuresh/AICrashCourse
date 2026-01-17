import os
import asyncio
import streamlit as st
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
# CONFIG
# ============================

TESTPLAN_FILE = "MultiMagixTestPlan.txt"
CODE_FILE = "MMMSSpeed.cpp"

TESTPLAN_DIR = "data/testplan"
CODE_DIR = "data/code"

os.makedirs(TESTPLAN_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)


# ============================
# INIT AGENT (run once)
# ============================

@st.cache_resource
def init_agent():

    with open('../keys/.openai.txt') as f:
        OPENAI_API_KEY = SecretStr(f.read().strip())

    # -------- Load docs --------
    testplan_docs = TextLoader(TESTPLAN_FILE, encoding="utf-8").load()
    code_docs = TextLoader(CODE_FILE, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    testplan_chunks = splitter.split_documents(testplan_docs)
    code_chunks = splitter.split_documents(code_docs)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )

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

    if testplan_vs._collection.count() == 0:
        testplan_vs.add_documents(testplan_chunks)

    if code_vs._collection.count() == 0:
        code_vs.add_documents(code_chunks)

    testplan_retriever = testplan_vs.as_retriever(search_kwargs={"k": 5})
    code_retriever = code_vs.as_retriever(search_kwargs={"k": 5})

    # -------- Prompts --------
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

    # -------- Tools --------
    @tool
    def ask_testplan(question: str) -> str:
        """Questions about test cases, validation, workflows, commands."""
        docs = testplan_retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        return testplan_chain.invoke({
            "question": question,
            "context": context
        })

    @tool
    def ask_codebase(question: str) -> str:
        """Questions about C++ code, functions, logic, bugs."""
        docs = code_retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        return code_chain.invoke({
            "question": question,
            "context": context
        })

    SYSTEM_PROMPT = """
You are a Software Development Assistant.

You have two knowledge sources:
1. Test Plan (testing, validation, commands, workflows)
2. Source Code (C++ implementation, functions, logic, bugs)

Routing rules:
- Testing/commands/workflows → ask_testplan
- Code/logic/functions → ask_codebase
- Always use one tool.
"""

    agent = create_agent(
        model=llm,
        tools=[ask_testplan, ask_codebase],
        system_prompt=SYSTEM_PROMPT
    )

    return agent


agent = init_agent()


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(page_title="🧠 Software Dev Assistant", page_icon="💻", layout="wide")

st.title("🧠 Software Dev Assistant")
st.caption("Agentic Multi-RAG over Test Plan + Source Code")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Ask about test plan or source code...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning and routing..."):
            response = agent.invoke({
                "messages": [HumanMessage(content=prompt)]
            })

            answer = response["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

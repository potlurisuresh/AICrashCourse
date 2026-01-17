import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr
from bs4.filter import SoupStrainer
from langchain_core.documents import Document


# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Maigha RAG", page_icon="🌐", layout="wide")
st.title("🌐 Maigha – Web RAG Assistant")
st.caption("Pure RAG system (no memory, no agents)")


# ---------------------------------
# Load API Key
# ---------------------------------
with open("../keys/.openai.txt") as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())


# ---------------------------------
# URLs
# ---------------------------------
maigha_page_url = "https://maigha.com/"
multimagix_page_url = "https://multimagix.com/"
upsilon_url = "https://www.upsilonservices.com/"

persist_dir = "data/maigha_knowledge_db"
os.makedirs(persist_dir, exist_ok=True)


# ---------------------------------
# Vector DB Builder (cached)
# ---------------------------------
@st.cache_resource(show_spinner="Building Maigha knowledge base...")
def build_vectorstores():

    # -------- Load full web pages --------
    loader_web = WebBaseLoader(
        web_paths=[maigha_page_url, multimagix_page_url, upsilon_url]
    )
    docs_web = loader_web.load()

    web_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    web_chunks = web_splitter.split_documents(docs_web)

    # -------- Scrape only features --------
    loader_web_scraping = WebBaseLoader(
        web_paths=[multimagix_page_url],
        bs_kwargs={
            "parse_only": SoupStrainer(class_="et_pb_module_header"),
        },
        bs_get_text_kwargs={"separator": " | ", "strip": True}
    )

    docs_scraped = loader_web_scraping.load()
    raw_text = docs_scraped[0].page_content

    remove_phrases = [
        "User-friendly tools",
        "Global accessibility",
        "User-Centric Design",
        "Versatility"
    ]

    clean_text = raw_text
    for phrase in remove_phrases:
        clean_text = clean_text.replace(phrase, "")

    clean_text = clean_text.replace("||", "|").strip(" |")
    features_text = "MultiMagix features: " + clean_text.replace("|", ",")

    doc = Document(
        page_content=features_text,
        metadata={"source": "multimagix_features"}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=60,
        chunk_overlap=10
    )

    feature_chunks = splitter.split_documents([doc])

    # -------- Embeddings --------
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )

    # -------- Vector stores --------
    features_vs = Chroma(
        collection_name="multimagix_features",
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    features_vs.add_documents(feature_chunks)

    web_vs = Chroma(
        collection_name="web_pages",
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    web_vs.add_documents(web_chunks)

    return features_vs.as_retriever(k=5), web_vs.as_retriever(k=5)


features_retriever, web_retriever = build_vectorstores()


# ---------------------------------
# Prompt + LLM
# ---------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant"),
    ("system", "Context:\n{context}"),
    ("user", "{question}")
])

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

chain = prompt_template | llm | StrOutputParser()


# ---------------------------------
# Session chat history (UI only)
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------
# Chat input
# ---------------------------------
user_input = st.chat_input("Ask about Maigha / MultiMagix / Upsilon...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    q = user_input.lower()

    # -------- Routing --------
    if any(word in q for word in ["feature", "features"]):
        retriever = features_retriever
        source = "multimagix_features"
    else:
        retriever = web_retriever
        source = "web_pages"

    with st.spinner("Searching knowledge base..."):
        docs = retriever.invoke(user_input)

    context = "\n\n".join(d.page_content for d in docs)

    answer = chain.invoke({
        "question": user_input,
        "context": context
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

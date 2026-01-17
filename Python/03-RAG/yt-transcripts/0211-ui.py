import os
import re
import shutil
from typing import Dict, Any

import yt_dlp
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎬", layout="wide")
st.title("🎬 YouTube RAG Assistant")
st.caption("Robust multi-video RAG with auto-rebuild")

# ---------------------------------
# API key
# ---------------------------------
with open("../keys/.openai.txt") as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# ---------------------------------
# YouTube URLs
# ---------------------------------
YOUTUBE_URLS = [
    "https://www.youtube.com/live/Epm2sP0iTpQ?si=3zQiY7Vkc3RfNTPr",
    "https://youtu.be/fiz56MShXb0?si=nn5PY4R8QBUdXdqi",
    "https://youtu.be/bOAisL_4MqQ?si=5forGR5JTslSu35h"
]

BASE_DIR = "rag_data/youtube"

# ---------------------------------
# Helpers
# ---------------------------------
def extract_video_id(url: str) -> str:
    patterns = [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"live/([^?]+)"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Invalid YouTube URL")

def get_video_title(url: str) -> str:
    ydl_opts: Dict[str, Any] = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: Dict[str, Any] = ydl.extract_info(url, download=False)
            title = info.get("title")
            if isinstance(title, str):
                return title
            return "Unknown Title"
    except Exception:
        return "Unknown Title"

# ---------------------------------
# Core components
# ---------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Answer only from the YouTube transcript."),
    ("system", "Transcript context:\n{context}"),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# ---------------------------------
# Robust vectorstore loader
# ---------------------------------
def load_or_rebuild_collection(collection_name: str, url: str):
    path = os.path.join(BASE_DIR, collection_name)
    os.makedirs(path, exist_ok=True)

    try:
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=path,
            embedding_function=embeddings
        )
        count = vs._collection.count()
        return vs, count
    except Exception:
        # corrupted DB
        st.warning(f"⚠️ Corrupted index detected for {collection_name}. Rebuilding...")
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        vs = Chroma(
            collection_name=collection_name,
            persist_directory=path,
            embedding_function=embeddings
        )
        return vs, 0

# ---------------------------------
# Build / load all video DBs
# ---------------------------------
@st.cache_resource(show_spinner=True)
def prepare_video_collections():
    video_map = {}

    for url in YOUTUBE_URLS:
        video_id = extract_video_id(url)
        collection_name = f"yt_{video_id}"
        title = get_video_title(url)

        vs, count = load_or_rebuild_collection(collection_name, url)

        if count == 0:
            loader = YoutubeLoader.from_youtube_url(url)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            vs.add_documents(chunks)

        video_map[title] = {
            "collection": collection_name,
            "path": os.path.join(BASE_DIR, collection_name)
        }

    return video_map

with st.spinner("Preparing YouTube knowledge bases..."):
    video_map = prepare_video_collections()

# ---------------------------------
# UI – video selector
# ---------------------------------
video_titles = list(video_map.keys())
selected_title = st.selectbox("🎥 Select a YouTube video", video_titles)

selected = video_map[selected_title]

vectorstore = Chroma(
    collection_name=selected["collection"],
    persist_directory=selected["path"],
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

st.success(f"Loaded: {selected_title}")
st.divider()

# ---------------------------------
# Chat state
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------
# Chat input
# ---------------------------------
question = st.chat_input("Ask something about this video...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        answer = chain.invoke({"question": question, "context": context})

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)


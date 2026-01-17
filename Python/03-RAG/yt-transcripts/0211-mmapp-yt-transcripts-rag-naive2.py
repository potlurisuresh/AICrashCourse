import re
import yt_dlp

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr

# ----------------------------
# API key
# ----------------------------
with open('../keys/.openai.txt') as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# ----------------------------
# YouTube URLs
# ----------------------------
YOUTUBE_URLS = [
    "https://www.youtube.com/live/Epm2sP0iTpQ?si=3zQiY7Vkc3RfNTPr",
    "https://youtu.be/fiz56MShXb0?si=nn5PY4R8QBUdXdqi",
    "https://youtu.be/bOAisL_4MqQ?si=5forGR5JTslSu35h"
]

# ----------------------------
# Helpers
# ----------------------------
def extract_video_id(url: str) -> str:
    patterns = [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"live/([^?]+)"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Invalid YouTube URL")

def get_video_title(url: str) -> str:
    ydl_opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
            info = ydl.extract_info(url, download=False)
            return info.get("title", "Unknown Title") # type: ignore
    except Exception:
        return "Unknown Title"

# ----------------------------
# Text splitter
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# ----------------------------
# Embeddings
# ----------------------------
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# ----------------------------
# Build vector stores (one per video)
# ----------------------------
video_map = {}

print("\n🔹 Loading / building YouTube collections...\n")

for idx, url in enumerate(YOUTUBE_URLS, start=1):
    video_id = extract_video_id(url)
    collection_name = f"yt_{video_id}"
    title = get_video_title(url)

    print(f"[{idx}] {title}")

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory="rag_data/youtube",
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        print("   → Loading transcript...")
        loader = YoutubeLoader.from_youtube_url(url)
        docs = loader.load()

        print("   → Splitting...")
        chunks = splitter.split_documents(docs)

        print("   → Storing embeddings...")
        vectorstore.add_documents(chunks)
        print("   → Done.\n")
    else:
        print("   → Already exists. Skipping embedding.\n")

    video_map[str(idx)] = {
        "title": title,
        "collection": collection_name
    }

# ----------------------------
# RAG chain
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Answer only from the YouTube transcript."),
    ("system", "Transcript context:\n{context}"),
    ("user", "{question}")
])

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

chain = prompt | llm | StrOutputParser()

# ----------------------------
# Ask questions
# ----------------------------
while True:

    print("\n==============================")
    print("🎬 Available Videos")
    print("==============================")

    for k, v in video_map.items():
        print(f"{k}. {v['title']}")

    choice = input("\nSelect a video number (or 'exit'): ").strip()
    if choice.lower() == "exit":
        break
    if choice not in video_map:
        print("❌ Invalid choice")
        continue

    selected = video_map[choice]

    vectorstore = Chroma(
        collection_name=selected["collection"],
        persist_directory="rag_data/youtube",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print(f"\n✅ Loaded: {selected['title']}")
    print("Ask questions (type 'back' to change video)")

    while True:
        user_input = input("\nAsk about this video: ").strip()
        if user_input.lower() == "back":
            break

        docs = retriever.invoke(user_input)
        context = "\n\n".join(d.page_content for d in docs)

        answer = chain.invoke({
            "question": user_input,
            "context": context
        })

        print("\n📺 RAG Answer:\n", answer)

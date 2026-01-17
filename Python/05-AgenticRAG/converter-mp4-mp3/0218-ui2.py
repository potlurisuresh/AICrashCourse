import streamlit as st
import os
import subprocess

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# CONFIG
# ============================================================
MMAPP_PATH = r"C:\\MMWS\\MMMediaSuite\\MMApp\\Console\\Windows\\x64\\Debug\\MMApp.exe"
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"
UPLOAD_DIR = "uploads"
PERSIST_DIR = "rag_data/testplan"
TESTPLAN_PATH = "MultiMagixTestPlan.txt"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# LOAD API KEY
# ============================================================
f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())
f.close()

# ============================================================
# BUILD / LOAD RAG (ONLY ONCE)
# ============================================================
@st.cache_resource
def load_rag():

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY, 
        model="text-embedding-3-large"
    )

    vectorstore = Chroma(
        collection_name="maigha_testplan_memory",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        st.write("📌 Indexing MultiMagix test plan (first run only)...")

        loader = TextLoader(TESTPLAN_PATH, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        chunks = splitter.split_documents(docs)

        vectorstore.add_documents(chunks)

        st.write("✅ Test plan indexed.")
    else:
        st.write("⚡ Loaded existing test plan memory.")

    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = load_rag()

# ============================================================
# TOOLS
# ============================================================
@tool
def search_testplan(query: str) -> str:
    """
    Search the MultiMagix test plan.
    """
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


@tool
def mp4_to_mp3(command_to_convert: str) -> str:
    """
    Execute mp4 to mp3 command via MMApp.
    """

    # Clear output folder
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    try:
        result = subprocess.run(
            command_to_convert,
            capture_output=True,
            text=True,
            shell=False
        )

        if result.returncode != 0:
            return f"❌ MMApp failed:\n{result.stderr}"

        mp3_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]

        if not mp3_files:
            return "❌ No MP3 generated."

        return f"SUCCESS::{os.path.join(OUTPUT_DIR, mp3_files[0])}"

    except Exception as e:
        return f"❌ Error running MMApp: {str(e)}"


tools = [search_testplan, mp4_to_mp3]

# ============================================================
# LLM + AGENT
# ============================================================
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0
)

SYSTEM_PROMPT = f"""
You are a MultiMagix test engineer assistant.

Rules:
1. If the user asks about commands, test steps, or workflows → use search_testplan.
2. If the user asks to convert mp4 to mp3:
   - First call search_testplan to find the correct command.
   - Then construct the full MMApp command using:
        MMAPP_PATH = {MMAPP_PATH}
        OUTPUT_DIR = {OUTPUT_DIR}
        UPLOADED_FILE path from user.
   - Then call mp4_to_mp3.
3. Never invent commands. Always fetch from the test plan.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="MultiMagix AI Assistant", page_icon="🎬", layout="centered")

st.title("🎬 MultiMagix AI Assistant")
st.caption("Agentic RAG + TestPlan Memory + Real MMApp Execution")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_mp3" not in st.session_state:
    st.session_state.last_mp3 = None

# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📤 Upload MP4")
uploaded_file = st.file_uploader("Upload MP4 file", type=["mp4"])

if uploaded_file:
    upload_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded: {uploaded_file.name}")
    st.session_state.upload_path = upload_path

# ----------------------------
# Chat History
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat Input
# ----------------------------
user_input = st.chat_input("Ask about test plan or say: Convert my uploaded video to mp3")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    if "upload_path" in st.session_state:
        lc_messages.append(
            HumanMessage(content=f"Uploaded file path: {st.session_state.upload_path}")
        )

    response = agent.invoke({"messages": lc_messages})

    # ✅ FIX: extract tool output properly
    ai_text = ""
    mp3_path = None

    for msg in response["messages"]:
        if isinstance(msg, AIMessage):
            ai_text = msg.content

        if isinstance(msg.content, str) and msg.content.startswith("SUCCESS::"):
            mp3_path = msg.content.replace("SUCCESS::", "").strip()
            st.session_state.last_mp3 = mp3_path

    if mp3_path:
        ai_text = "✅ Conversion successful. MP3 generated."

    st.session_state.messages.append({"role": "assistant", "content": ai_text})

    with st.chat_message("assistant"):
        st.markdown(ai_text)

# ----------------------------
# Download Section
# ----------------------------
if st.session_state.last_mp3 and os.path.exists(st.session_state.last_mp3):

    st.subheader("⬇️ Download Output")

    with open(st.session_state.last_mp3, "rb") as f:
        st.download_button(
            label="Download MP3",
            data=f,
            file_name=os.path.basename(st.session_state.last_mp3),
            mime="audio/mpeg"
        )

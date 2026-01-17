import streamlit as st
import os
import shutil
import subprocess

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------
# CONFIG
# ----------------------------
MMAPP_PATH = r"C:\\MMWS\\MMMediaSuite\\MMApp\\Console\\Windows\\x64\\Debug\\MMApp.exe"
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Load API Key
# ----------------------------
f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())
f.close()

# ----------------------------
# Tool
# ----------------------------
@tool
def mp4_to_mp3(input_file: str) -> str:
    """
    Convert an MP4 file to MP3 using MMApp.exe.
    Returns the output MP3 file path.
    """

    if not os.path.exists(input_file):
        return f"❌ Input file not found: {input_file}"

    # Clear output folder
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    cmd = [
        MMAPP_PATH,
        "-f", "1",
        "-i", input_file,
        "-o", "1"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return f"❌ MMApp failed:\n{result.stderr}"

        mp3_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]

        if not mp3_files:
            return "❌ No MP3 file generated."

        output_file = os.path.join(OUTPUT_DIR, mp3_files[0])
        return f"SUCCESS::{output_file}"

    except Exception as e:
        return f"❌ Error running MMApp: {str(e)}"


# ----------------------------
# LLM + Agent
# ----------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0
)

SYSTEM_PROMPT = """
You are a media assistant.
If the user asks to convert video to audio (mp4 to mp3), use the mp4_to_mp3 tool.
Otherwise, answer normally.
"""

agent = create_agent(
    model=llm,
    tools=[mp4_to_mp3],
    system_prompt=SYSTEM_PROMPT
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="MultiMagix Media Agent", page_icon="🎬", layout="centered")
st.title("🎬 MultiMagix Media Agent")
st.caption("Agent + MMApp.exe (MP4 → MP3)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_mp3" not in st.session_state:
    st.session_state.last_mp3 = None

# ----------------------------
# File Upload
# ----------------------------
st.subheader("📤 Upload MP4 File")
uploaded_file = st.file_uploader("Upload an MP4 file", type=["mp4"])

if uploaded_file:
    upload_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded: {uploaded_file.name}")
    st.session_state.upload_path = upload_path

# ----------------------------
# Show chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Example: Convert my uploaded video to mp3")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Build LC messages
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # If file uploaded, inject file path into prompt
    if "upload_path" in st.session_state:
        lc_messages.append(
            HumanMessage(content=f"Uploaded file path: {st.session_state.upload_path}")
        )

    # Call agent
    response = agent.invoke({"messages": lc_messages})
    ai_text = response["messages"][-1].content

    mp3_path = None
    if ai_text.startswith("SUCCESS::"):
        mp3_path = ai_text.replace("SUCCESS::", "").strip()
        ai_text = "✅ Conversion successful. MP3 file generated."

    st.session_state.messages.append({"role": "assistant", "content": ai_text})

    with st.chat_message("assistant"):
        st.markdown(ai_text)

        if mp3_path and os.path.exists(mp3_path):
            st.session_state.last_mp3 = mp3_path

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

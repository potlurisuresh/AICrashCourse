import asyncio
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


UPLOAD_DIR = "uploads"
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------
# Init MCP Agent (runs once)
# -----------------------------------
@st.cache_resource
def init_agent():

    f = open('../keys/.openai.txt')
    OPENAI_API_KEY = SecretStr(f.read().strip())
    f.close()

    client = MultiServerMCPClient(
        {
            "media": {
                "transport": "sse",
                "url": "http://127.0.0.1:8000/sse"
            }
        }
    )

    async def _init():
        tools = await client.get_tools()

        SYSTEM_PROMPT = f"""
You are a helpful AI media assistant.

Rules:
- If user asks to convert mp4 to mp3, extract audio, or create mp3 from video
  you MUST use mp4_to_mp3 tool.
- All output files must be written into this folder: {OUTPUT_DIR}
- Always return the final output file path clearly.

Example:
Input: uploads/sample.mp4
Output: outputs/sample.mp3
"""

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4.1-mini",
            temperature=0.0,
            max_completion_tokens=300
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT
        )

        return agent

    return asyncio.run(_init())


agent = init_agent()

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="🎬 Maigha Media MCP Agent", page_icon="🎧", layout="wide")

st.title("🎬 Maigha Media MCP Agent")
st.caption("Upload → Agent → MCP Server → Download")

# -----------------------------
# File upload section
# -----------------------------
st.subheader("📤 Upload Media File")

uploaded_file = st.file_uploader("Upload mp4 / mov / mkv", type=["mp4", "mov", "mkv", "avi"])

uploaded_path = None

if uploaded_file:
    uploaded_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded: {uploaded_path}")

# -----------------------------
# Chat section
# -----------------------------
st.subheader("💬 Media Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Example: Convert uploaded file to mp3")

if prompt:
    if uploaded_path:
        prompt = f"{prompt}\n\nInput file: {uploaded_path}"

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing via MCP server..."):
            response = asyncio.run(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            )

            ai_text = response["messages"][-1].content
            st.markdown(ai_text)

    st.session_state.messages.append({"role": "assistant", "content": ai_text})

# -----------------------------
# Download section
# -----------------------------
st.subheader("📥 Download Outputs")

files = os.listdir(OUTPUT_DIR)

if not files:
    st.info("No processed files yet.")
else:
    for file in files:
        file_path = os.path.join(OUTPUT_DIR, file)

        with open(file_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {file}",
                data=f,
                file_name=file,
                mime="application/octet-stream"
            )

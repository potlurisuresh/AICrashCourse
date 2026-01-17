import streamlit as st
import uuid
from PIL import Image
from huggingface_hub import InferenceClient

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------
# Load API Keys
# ----------------------------
f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())
f.close()

f = open('../keys/.hgface.txt')
HF_API_KEY = SecretStr(f.read().strip())
f.close()

# ----------------------------
# HuggingFace Client
# ----------------------------
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY.get_secret_value()
)

# ----------------------------
# Tool
# ----------------------------
@tool
def generate_image_hf(prompt: str) -> str:
    """
    Generate image using HuggingFace and save locally.
    Returns the saved file path.
    """

    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-dev"
    )

    filename = f"hf_image_{uuid.uuid4().hex[:8]}.png"
    image.save(filename)

    return filename

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0
)

# ----------------------------
# Agent
# ----------------------------
agent = create_agent(
    model=llm,
    tools=[generate_image_hf],
    system_prompt="You are an assistant that can generate images when required."
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Maigha Image Agent", page_icon="🎨", layout="centered")
st.title("🎨 Maigha Image Agent")
st.caption("Agent → Tool → Image → Save → Display")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_path"):
            st.image(msg["image_path"], use_container_width=True)

# Input
user_input = st.chat_input("Describe the image you want...")

if user_input:

    # Save user msg
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert to LC format
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Call agent
    response = agent.invoke({"messages": lc_messages})

    ai_text = response["messages"][-1].content

    # If tool returned filename, show image
    image_path = None
    if ai_text.endswith(".png"):
        image_path = ai_text

    # Save assistant msg
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_text,
        "image_path": image_path
    })

    with st.chat_message("assistant"):
        st.markdown(ai_text)
        if image_path:
            st.image(image_path, caption="Generated Image", use_container_width=True)

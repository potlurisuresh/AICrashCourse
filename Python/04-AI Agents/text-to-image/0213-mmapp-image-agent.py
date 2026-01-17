from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from huggingface_hub import InferenceClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import uuid
import base64
from openai import OpenAI

f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())

f = open('../keys/.hgface.txt')
HF_API_KEY = SecretStr(f.read().strip())

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY.get_secret_value()
)

SYSTEM_PROMPT = """You are a helpful AI assistant.
You have access to tools.
Use tools whenever real-world or real-time information is required.
If a tool is not required, answer directly.
"""

@tool
def generate_image_hf(prompt: str) -> str:
    """
    Generate an image using Hugging Face Inference API and save it locally.
    """

    image = client.text_to_image(
        prompt=prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0"
    )

    filename = f"hf_image_{uuid.uuid4().hex[:8]}.png"
    image.save(filename)

    return f"✅ HuggingFace image generated: {filename}"


llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                model="gpt-4.1-mini", 
                temperature=0.0,
                max_completion_tokens=300)

agent = create_agent(
    model=llm,
    tools=[generate_image_hf],
    system_prompt=SYSTEM_PROMPT
)

while True:
    user_input = input("\nMaigha: ").strip()
    if user_input.lower() == "exit":
        break

    response = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    for msg in response["messages"]:
        msg.pretty_print()

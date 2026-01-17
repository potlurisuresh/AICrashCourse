from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from datetime import datetime
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
import subprocess
import os

f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())

MMAPP_PATH = r"C:\\MMWS\\MMMediaSuite\\MMApp\\Console\\Windows\\x64\Debug\\MMApp.exe"
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"

@tool
def mp4_to_mp3(input_file: str) -> str:
    """
    Convert an MP4 file to MP3 using MMApp.exe.
    Returns the output MP3 file path.
    """

    if not os.path.exists(input_file):
        return f"❌ Input file not found: {input_file}"
    
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    cmd = [
        MMAPP_PATH,
        "-f", "1",
        "-i", input_file,
        "-o", "1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False
        )

        if result.returncode != 0:
            return f"❌ MMApp failed:\n{result.stderr}"

        mp3_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]

        if mp3_files:
            for f in mp3_files:
                output_file = f
        else:
            print("❌ No MP3 files found in the folder.")

        return f"✅ Conversion successful.\nMP3 saved at:\n{output_file}"

    except Exception as e:
        return f"❌ Error running MMApp: {str(e)}"


SYSTEM_PROMPT = """
You are a media assistant.
If the user asks to convert video to audio (mp4 to mp3), use the mp4_to_mp3 tool.
Otherwise, answer normally.
"""


llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                model="gpt-4.1-mini", 
                temperature=0.0,
                max_completion_tokens=300)

agent = create_agent(
    model=llm,
    tools=[mp4_to_mp3],
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

import subprocess

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())

MMAPP_PATH = r"C:\\MMWS\\MMMediaSuite\\MMApp\\Console\\Windows\\x64\Debug\\"

####RAG#########
loader = TextLoader("MultiMagixTestPlan.txt", encoding="utf-8")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY, 
    model="text-embedding-3-large"
)

persist_dir = "rag_data/testplan"

vectorstore = Chroma(
    collection_name="maigha_testplan_memory",
    persist_directory=persist_dir,
    embedding_function=embeddings
)

vectorstore.add_documents(chunks)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

#### TOOLS ####

@tool
def search_testplan(query: str) -> str:
    """
    Search the MultiMagix test plan.

    Use this tool ONLY when the user asks about:
    - test plan
    - test cases
    - commands
    - validation steps
    - workflows
    - how to use MMApp

    This tool returns relevant sections of the test plan.
    It DOES NOT execute anything.
    """
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def mp4_to_mp3(command_to_convert: str) -> str:
    """
    Execute an MP4 to MP3 conversion command.

    Use this tool ONLY when you already have a concrete command
    (usually obtained from the test plan via search_testplan).

    Input must be a full command line string.

    Example input:
    MMApp.exe -i input.mp4 -o output.mp3

    This tool prints the command and simulates execution.
    It returns "OK" after execution.
    """
    print(command_to_convert)
    
    try:
        result = subprocess.run(
            MMAPP_PATH + command_to_convert,
            capture_output=True,
            text=True,
            shell=False
        )

        if result.returncode != 0:
            return f"❌ MMApp failed:\n{result.stderr}"

        return f"✅ Conversion successful.\n"

    except Exception as e:
        return f"❌ Error running MMApp: {str(e)}"


llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                model="gpt-4.1-mini", 
                temperature=0.0)


tools = [search_testplan, mp4_to_mp3]


SYSTEM_PROMPT = """
You are a MultiMagix test engineer assistant.

Your job is to:
1. Understand the user request.
2. If the user asks about test plan, commands, workflows, or validation steps:
   → Call search_testplan.

3. If the user asks to convert media (mp4 to mp3):
   → First call search_testplan to find the correct command.
   → Then call mp4_to_mp3 with the extracted command.

4. If the question is general, answer directly.

Never invent commands. Always fetch them from the test plan.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT)

while True:
    q = input("\nAsk MultiMagix Agent (or 'exit'): ").strip()
    if q.lower() == "exit":
        break

    response = agent.invoke({
        "messages": [HumanMessage(content=q)]
    })

    for msg in response["messages"]:
        msg.pretty_print()


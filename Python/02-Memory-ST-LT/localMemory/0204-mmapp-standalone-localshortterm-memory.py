from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load API key
with open("../keys/.openai.txt") as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical assistant"),
    ("human", "{topic}")
])

# LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_completion_tokens=300
)

chain = prompt | llm | StrOutputParser()

# ---- In-memory chat history (short-term memory) ----
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="topic"
)

# ---- Chat loop ----
print("Type 'exit' to quit\n")

while True:
    user_input = input("Ask Maigha: ").strip()
    if user_input.lower() == "exit":
        break

    output = chain_with_memory.invoke(
        {"topic": user_input},
        config={"configurable": {"session_id": "maigha-session"}}
    )

    print("\nMaigha:", output, "\n")

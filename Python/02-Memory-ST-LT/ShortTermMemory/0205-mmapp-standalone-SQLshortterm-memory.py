# ! pip install langchain-community SQLAlchemy

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
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

# ----------------------------
# DB setup
# ----------------------------
db_path = "chat_data/sql_chat_history.db"
engine = create_engine(f"sqlite:///{db_path}")

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id,connection=engine)

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

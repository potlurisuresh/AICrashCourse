from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma

# Load API key
with open("../keys/.openai.txt") as f:
    OPENAI_API_KEY = SecretStr(f.read().strip())

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical assistant"),
    ("system", "Relevant long term memory:\n{long_memory}"),
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
# DB setup short term
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

# ----------------------------
# Long-term memory (Chroma)
# ----------------------------
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY, 
    model="text-embedding-3-large"
)

persist_dir = "chat_data/long_term_memory"

vectorstore = Chroma(
    collection_name="maigha_memory",
    persist_directory=persist_dir,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def save_long_term(text, session_id):
    vectorstore.add_texts([f"[{session_id}] {text}"])

def recall_long_term(query):
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

# ---- Chat loop ----
print("Type 'exit' to quit\n")

SESSION_ID = "maigha-session"

while True:
    user_input = input("Ask Maigha: ").strip()
    if user_input.lower() == "exit":
        break
   
    long_memory = recall_long_term(user_input)

    output = chain_with_memory.invoke(
        {"topic": user_input, "long_memory": long_memory},
        config={"configurable": {"session_id": "maigha-session"}}
    )

    # Save to long-term memory
    save_long_term(user_input, SESSION_ID)
    save_long_term(output, SESSION_ID)

    print("\nMaigha:", output, "\n")

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical assistant"),
    ("user", "{topic}")
])

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                model="gpt-4.1-mini", 
                temperature=0.0,
                max_completion_tokens=300)

output_parser = StrOutputParser()

chain = prompt_template | llm | output_parser

user_input = input("Ask Maigha: ").strip()
user_prompt = {"topic": user_input}

output = chain.invoke(user_prompt)

print(output)

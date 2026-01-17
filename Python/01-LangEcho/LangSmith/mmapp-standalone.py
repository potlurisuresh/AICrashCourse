import os

# Setup LANGSMITH API Key
f = open('../keys/.langsmith_api_key.txt')
LANGSMITH_API_KEY = f.read()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = "maigha-ai-empowerment-jan16"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display


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


# -----------------------------
# State Definition
# -----------------------------
class ChatState(TypedDict):
    userinput: str
    response: str

# -----------------------------
# Graph Node
# -----------------------------
def llm_node(state: ChatState) -> ChatState:
    output = chain.invoke({"topic": state["userinput"]})
    state["response"] = output
    return state

# -----------------------------
# Build Graph
# -----------------------------
workflow = StateGraph(ChatState)

workflow.add_node("llm_node", llm_node)

workflow.add_edge(START, "llm_node")
workflow.add_edge("llm_node", END)

# -----------------------------
# Run Graph
# -----------------------------
graph = workflow.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))  #works in jupyter notebook

### INVOKE GRAPH
user_input = input("Ask Maigha: ").strip()
result = graph.invoke({"userinput": user_input, "response": ""})

print(result["response"])

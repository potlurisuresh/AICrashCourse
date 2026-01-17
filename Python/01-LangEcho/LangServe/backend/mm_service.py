from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda

# -----------------------------
# State Definition
# -----------------------------
class ChatState(TypedDict):
    userinput: str
    response: str


# -----------------------------
# Module-level shared graph
# -----------------------------
_graph: Optional[Runnable] = None


def init_llm(key_path: str = "keys/.openai.txt") -> None:
    """
    Initialize LangGraph ONCE.
    Safe to call multiple times.
    """
    global _graph

    if _graph is not None:
        return  # already initialized

    # Load API key
    with open(key_path, "r") as f:
        api_key = SecretStr(f.read().strip())

    # Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a precise technical assistant"),
        ("user", "{topic}")
    ])

    # LLM
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4.1-mini",
        temperature=0.0,
        max_completion_tokens=300
    )

    # Output parser
    output_parser = StrOutputParser()

    # Chain
    chain = prompt_template | llm | output_parser

    # -----------------------------
    # Graph Node
    # -----------------------------
    def llm_node(state: ChatState) -> ChatState:
        output = chain.invoke({"topic": state["userinput"]})
        return {
            "userinput": state["userinput"],
            "response": output
        }

    # -----------------------------
    # Build Graph
    # -----------------------------
    workflow = StateGraph(ChatState)

    workflow.add_node("llm_node", llm_node)
    workflow.add_edge(START, "llm_node")
    workflow.add_edge("llm_node", END)

    _graph = workflow.compile()


def generate_response(prompt: str) -> str:
    """
    Run inference using the initialized LangGraph.
    """
    if _graph is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")

    result = _graph.invoke({
        "userinput": prompt,
        "response": ""
    })

    return result["response"]

# -----------------------------
# LangChain Runnable
# -----------------------------
llm_runnable = RunnableLambda(generate_response)

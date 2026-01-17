from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from langchain_core.runnables import Runnable
from typing import Optional

# -----------------------------
# Module-level shared objects
# -----------------------------
_chain: Optional[Runnable] | None = None


def init_llm(key_path: str = "../keys/.openai.txt") -> None:
    """
    Initialize LangChain prompt, model, parser, and chain ONCE.
    Safe to call multiple times.
    """
    global _chain

    if _chain is not None:
        return  # already initialized

    # Load API key
    with open(key_path, "r") as f:
        api_key = SecretStr(f.read().strip())

    # Prompt template
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

    # Build chain ONCE
    _chain = prompt_template | llm | output_parser


def generate_response(prompt: str) -> str:
    """
    Generate response using the initialized LangChain chain.
    """
    if _chain is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")

    result = _chain.invoke({"topic": prompt})
    return result

from fastapi import FastAPI
from langserve import add_routes

from mm_service import init_llm, llm_runnable

# Initialize OpenAI ONCE at startup
init_llm()

multimagix = FastAPI(
    title="MultiMagix LLM API",
    version="1.0",
    description="LangServe deployment for MultiMagix"
)

# Add LangServe routes
add_routes(
    multimagix,
    llm_runnable,
    path="/llm"
)

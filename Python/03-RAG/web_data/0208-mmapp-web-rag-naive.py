from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr
from bs4.filter import SoupStrainer
from langchain_core.documents import Document

f = open('../keys/.openai.txt')
OPENAI_API_KEY = SecretStr(f.read().strip())

# ----------------------------
# 1. Load webpage (RAG source) and split
# ----------------------------
maigha_page_url = "https://maigha.com/"
multimagix_page_url = "https://multimagix.com/"
upsilon_url = "https://www.upsilonservices.com/"

loader_web = WebBaseLoader(web_paths=[maigha_page_url, multimagix_page_url, upsilon_url])
docs_web = loader_web.load()

web_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

web_chunks = web_splitter.split_documents(docs_web)

# =======================================================================
loader_web_scraping = WebBaseLoader(
    web_paths=[multimagix_page_url],
    bs_kwargs={
        "parse_only": SoupStrainer(class_="et_pb_module_header"),
    },
    bs_get_text_kwargs={"separator": " | ", "strip": True}
)
docs_scraped = loader_web_scraping.load()

raw_text = docs_scraped[0].page_content

remove_phrases = [
    "User-friendly tools",
    "Global accessibility",
    "User-Centric Design",
    "Versatility"
]

clean_text = raw_text

for phrase in remove_phrases:
    clean_text = clean_text.replace(phrase, "")

# Cleanup extra separators and spaces
clean_text = clean_text.replace("||", "|")
clean_text = clean_text.strip(" |")

features_text = "MultiMagix features: " + clean_text
features_text = features_text.replace("|", ",")

doc = Document(
    page_content=features_text,
    metadata={"source": "multimagix_features"}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10
)

feature_chunks = splitter.split_documents([doc])
# =====================================================


# ----------------------------
# 2. Create vector store
# ----------------------------
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY, 
    model="text-embedding-3-large"
)

persist_dir = "data/maigha_knowledge_db"
#-------------------------------------------

features_vs = Chroma(
    collection_name="multimagix_features",
    persist_directory=persist_dir,
    embedding_function=embeddings
)

features_vs.add_documents(feature_chunks)

features_retriever  = features_vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
#--------------------------------------------------
web_vs = Chroma(
    collection_name="web_pages",
    persist_directory=persist_dir,
    embedding_function=embeddings
)

web_vs.add_documents(web_chunks)

web_retriever = web_vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
# -------------------------------------------------


# ------- CHAIN -------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI assistant"),
    ("system", "Context:\n{context}"),
    ("user", "{question}")
])

llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                model="gpt-4.1-mini", 
                temperature=0.0,
                max_completion_tokens=300)

output_parser = StrOutputParser()

chain = prompt_template | llm | output_parser

while True:
    user_input = input("\nAsk Maigha (or 'exit'): ").strip()
    if user_input.lower() == "exit":
        break

    q = user_input.lower()

    # -------------------------
    # Simple routing logic
    # -------------------------
    if any(word in q for word in ["features"]):
        print("[Using collection: multimagix_features]")
        docs = features_retriever.invoke(user_input)
    else:
        print("[Using collection: web_pages]")
        docs = web_retriever.invoke(user_input)

    # -------------------------
    # Build context
    # -------------------------
    context = "\n\n".join(d.page_content for d in docs)

    # -------------------------
    # Ask LLM
    # -------------------------
    answer = chain.invoke({
        "question": user_input,
        "context": context
    })

    print("\nAnswer:\n", answer)



## Scrapping tools availble in marketing via API open-source
##    https://www.firecrawl.dev/

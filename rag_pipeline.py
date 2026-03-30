"""
rag_pipeline.py
Core RAG pipeline for the Research Paper Assistant.
Handles PDF loading, chunking, FAISS indexing, and QA chain construction.
"""
 
import os
from typing import List, Tuple, Optional
 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
 
# ── Constants ─────────────────────────────────────────────────────────────────
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
 
QA_PROMPT_TEMPLATE = """You are an expert research paper assistant. Your job is to help users understand academic papers clearly and thoroughly.
 
Using the context excerpts below from the research paper, answer the question in a helpful, detailed, and well-structured way.
 
Rules:
- Always answer based on the provided context
- If asked for a summary, provide a comprehensive overview covering: objective, methods, results, and conclusions
- Use bullet points when listing multiple items
- Be specific — include numbers, percentages, model names, dataset names when available
- If the context does not contain enough information, say what you DO know and mention what seems missing
- Never refuse to answer if there is any relevant information in the context
 
Context:
{context}
 
Question: {question}
 
Answer:"""
 
 
# ── PDF Loading & Chunking ────────────────────────────────────────────────────
def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Tuple[list, int]:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    num_pages = len(documents)
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks, num_pages
 
 
# ── FAISS Vector Store ────────────────────────────────────────────────────────
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
 
 
def build_faiss_index(chunks: list) -> FAISS:
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
 
 
def save_index(vectorstore: FAISS, path: str = INDEX_DIR):
    vectorstore.save_local(path)
 
 
def load_index(path: str = INDEX_DIR) -> Optional[FAISS]:
    # Optional[FAISS] instead of FAISS | None — compatible with Python 3.9
    if not os.path.exists(path):
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
 
 
# ── QA Chain ──────────────────────────────────────────────────────────────────
def build_qa_chain(
    vectorstore: FAISS,
    model: str = "llama-3.3-70b-versatile",
    top_k: int = 6,
):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
 
    llm = ChatOpenAI(
        model=model,
        openai_api_key=groq_api_key,
        openai_api_base=GROQ_BASE_URL,
        temperature=0.3,
        max_tokens=2048,
    )
 
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
 
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.7},
    )
 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
 
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
 
    def qa_chain(inputs: dict) -> dict:
        query = inputs["query"]
        source_docs = retriever.invoke(query)
        result = chain.invoke(query)
        return {"result": result, "source_documents": source_docs}
 
    return qa_chain
 
 

"""
utils.py
Utility helpers for the Research Paper Assistant.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple


# ── File Helpers ──────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to a temp path and return the path.
    Caller is responsible for deleting the file after use.
    """
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def ensure_dir(path: str):
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ── Text Helpers ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Basic text normalization:
      - Collapse multiple whitespace/newlines
      - Remove non-printable characters
    """
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)   # remove non-ASCII control chars
    text = re.sub(r"\n{3,}", "\n\n", text)           # max 2 consecutive newlines
    text = re.sub(r" {2,}", " ", text)               # collapse spaces
    return text.strip()


def truncate(text: str, max_chars: int = 300, ellipsis: str = "…") -> str:
    """Truncate text to max_chars and append ellipsis if needed."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + ellipsis


# ── Source Formatting ─────────────────────────────────────────────────────────
def format_sources(source_documents) -> List[dict]:
    """
    Convert LangChain source_documents into a list of dicts with:
      page, snippet, source (filename)
    """
    formatted = []
    seen_pages = set()
    for doc in source_documents:
        page = doc.metadata.get("page", "?")
        if page in seen_pages:
            continue
        seen_pages.add(page)
        formatted.append({
            "page": page,
            "snippet": truncate(doc.page_content, 200),
            "source": doc.metadata.get("source", "unknown"),
        })
    return formatted


# ── Suggested Questions ───────────────────────────────────────────────────────
GENERIC_QUESTIONS = [
    "What is the main research question or problem addressed?",
    "What methodology or approach was used?",
    "What are the key findings or results?",
    "What datasets or benchmarks were used?",
    "What are the limitations of this work?",
    "How does this compare to prior work?",
    "What future work do the authors suggest?",
    "What is the significance or impact of this research?",
]


def get_suggested_questions(paper_title: str = "") -> List[str]:
    """Return a list of suggested questions (generic if no title provided)."""
    return GENERIC_QUESTIONS


# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(chunks) -> dict:
    """Compute basic stats about the chunked document."""
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chunk = total_chars // len(chunks) if chunks else 0
    pages = sorted({c.metadata.get("page", 0) for c in chunks})
    return {
        "num_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_size": avg_chunk,
        "pages_covered": len(pages),
    }
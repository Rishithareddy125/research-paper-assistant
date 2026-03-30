import streamlit as st
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv

# ── Safe import ──────────────────────────────────────────────────────────────
try:
    from rag_pipeline import (
        load_and_chunk_pdf,
        build_faiss_index,
        build_qa_chain,
        save_index,
        load_index,
    )
    RAG_OK = True
    RAG_ERROR = None
except Exception as e:
    RAG_OK = False
    RAG_ERROR = str(e)

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="wide"
)

# ── Minimal styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0a0a0f; color: #e8e8f0; }
.chat-user { background:#7c6aff; padding:10px; border-radius:10px; color:white; margin-bottom:8px; }
.chat-ai { background:#13131f; padding:10px; border-radius:10px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Show backend errors ──────────────────────────────────────────────────────
if not RAG_OK:
    st.error("❌ Failed to load rag_pipeline")
    st.code(RAG_ERROR)
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("papers", {}),
    ("active_paper", None),
    ("conversations", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 PaperMind")

    if groq_api_key:
        st.success("API Key Loaded")
    else:
        st.error("Set GROQ_API_KEY in Streamlit Secrets")

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
    )

    top_k = st.slider("Top K", 2, 10, 5)
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # ── Process PDFs ─────────────────────────────────────────────────────────
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.papers:
                if not groq_api_key:
                    st.error("API key required")
                else:
                    with st.spinner(f"Processing {uf.name}..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uf.read())
                                tmp_path = tmp.name

                            chunks, pages = load_and_chunk_pdf(tmp_path, chunk_size, 50)
                            vs = build_faiss_index(chunks)
                            qa = build_qa_chain(vs, model, top_k)

                            st.session_state.papers[uf.name] = {
                                "vectorstore": vs,
                                "qa_chain": qa,
                                "chunks": len(chunks),
                                "pages": pages,
                            }

                            st.session_state.conversations[uf.name] = []
                            st.session_state.active_paper = uf.name

                            os.unlink(tmp_path)
                            st.success(f"{uf.name} ready!")

                        except Exception as e:
                            st.error(f"Error: {e}")

# ── Main area ────────────────────────────────────────────────────────────────
active = st.session_state.active_paper
paper_data = st.session_state.papers.get(active)
conversation = st.session_state.conversations.get(active, [])

# No paper
if not active:
    st.info("👈 Upload a PDF to start")
    st.stop()

# Header
st.title(f"📄 {active}")
st.caption(f"{paper_data['pages']} pages • {paper_data['chunks']} chunks")

# ── Chat display ─────────────────────────────────────────────────────────────
for msg in conversation:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>🧠 {msg['content']}</div>", unsafe_allow_html=True)

        if msg.get("sources"):
            st.caption(f"📖 Pages: {sorted(set(msg['sources']))}")

# ── Input ────────────────────────────────────────────────────────────────────
query = st.text_input("Ask something about the paper")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            # 🧠 Add memory (IMPORTANT UPGRADE)
            history = ""
            if len(conversation) >= 2:
                recent = conversation[-6:]
                for m in recent:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history += f"{role}: {m['content']}\n"

            full_query = query + "\n\n" + history if history else query

            result = paper_data["qa_chain"]({"query": full_query})
            answer = result["result"]

            sources = list(set(
                doc.metadata.get("page", "?")
                for doc in result.get("source_documents", [])
            ))

            now = datetime.now().strftime("%H:%M")

            conversation.append({
                "role": "user",
                "content": query,
                "time": now
            })

            conversation.append({
                "role": "ai",
                "content": answer,
                "sources": sources,
                "time": now
            })

            st.session_state.conversations[active] = conversation
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

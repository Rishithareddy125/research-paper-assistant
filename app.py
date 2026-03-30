import streamlit as st
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
 
load_dotenv()
 
# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PaperMind — Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
 
/* Global */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
 
/* Hide default Streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }
 
/* Sidebar */
[data-testid="stSidebar"] {
    background: #0e1117;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    margin-bottom: 0;
}
 
/* Paper list buttons */
.stButton > button {
    width: 100%;
    background: #1a1f2e;
    color: #94a3b8;
    border: 1px solid #2d3748;
    border-radius: 6px;
    font-size: 0.82rem;
    text-align: left;
    padding: 0.5rem 0.75rem;
    transition: all 0.2s;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.stButton > button:hover {
    background: #1e293b;
    color: #e2e8f0;
    border-color: #4f6af0;
}
 
/* Active paper button */
.active-paper > button {
    background: #1e3a5f !important;
    color: #93c5fd !important;
    border-color: #3b82f6 !important;
}
 
/* Chat bubbles */
.chat-user {
    background: #1e293b;
    border-left: 3px solid #4f6af0;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
    font-size: 0.92rem;
}
.chat-ai {
    background: #0f1923;
    border-left: 3px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: #d1fae5;
    font-size: 0.92rem;
    line-height: 1.6;
}
.chat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.6;
}
 
/* Paper stats badge */
.paper-stats {
    display: flex;
    gap: 1rem;
    margin: 0.5rem 0 1.5rem;
    flex-wrap: wrap;
}
.stat-badge {
    background: #1a2535;
    border: 1px solid #2d3748;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #64748b;
}
 
/* Title */
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #e2e8f0;
}
.app-subtitle {
    color: #475569;
    font-size: 0.9rem;
    margin-top: -0.5rem;
}
 
/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #374151;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
 
/* Divider */
hr { border-color: #1f2937; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Lazy import with helpful error ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    try:
        from rag_pipeline import load_and_chunk_pdf, build_faiss_index, build_qa_chain
        return load_and_chunk_pdf, build_faiss_index, build_qa_chain
    except ImportError as e:
        return None, None, None
 
 
# ── API Key ───────────────────────────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
 
# ── Session State ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "papers": {},
    "active_paper": None,
    "conversations": {},
    "processing_errors": {},
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
 
# ── Helper: truncate filename for display ─────────────────────────────────────
def truncate(name: str, max_len: int = 28) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"
 
 
# ── Helper: format timestamp ──────────────────────────────────────────────────
def fmt_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M")
    except Exception:
        return ""
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 PaperMind")
    st.caption("AI Research Assistant")
    st.divider()
 
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found.\nSet it in `.env` or Streamlit Secrets.")
 
    # ── Model settings ────────────────────────────────────────────────────────
    with st.expander("⚙️ Model Settings", expanded=False):
        model = st.selectbox(
            "LLM Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            help="Larger models are more accurate but slower.",
        )
        top_k = st.slider("Retrieved chunks (k)", min_value=2, max_value=10, value=5,
                          help="How many document chunks to retrieve per query.")
        chunk_size = st.slider("Chunk size (tokens)", min_value=200, max_value=1000,
                               value=500, step=50,
                               help="Larger chunks = more context, smaller = finer retrieval.")
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=200, value=50, step=10)
 
    st.divider()
 
    # ── File upload ───────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "📎 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers.",
    )
 
    if uploaded_files:
        load_and_chunk_pdf, build_faiss_index, build_qa_chain = get_rag_pipeline()
        if load_and_chunk_pdf is None:
            st.error("❌ `rag_pipeline.py` not found or has import errors.")
        elif not groq_api_key:
            st.warning("Add your GROQ_API_KEY to process papers.")
        else:
            for uf in uploaded_files:
                if uf.name in st.session_state.papers:
                    continue  # already indexed
                with st.spinner(f"Indexing {truncate(uf.name)}…"):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
 
                        chunks, num_pages = load_and_chunk_pdf(tmp_path, chunk_size, chunk_overlap)
                        vs = build_faiss_index(chunks)
                        qa = build_qa_chain(vs, model, top_k)
 
                        st.session_state.papers[uf.name] = {
                            "vectorstore": vs,
                            "qa_chain": qa,
                            "chunks": len(chunks),
                            "pages": num_pages,
                            "model": model,
                            "indexed_at": datetime.now().isoformat(),
                        }
                        st.session_state.conversations[uf.name] = []
 
                        if not st.session_state.active_paper:
                            st.session_state.active_paper = uf.name
 
                        st.session_state.processing_errors.pop(uf.name, None)
                        st.success(f"✅ {truncate(uf.name, 20)} ready")
 
                    except Exception as e:
                        st.session_state.processing_errors[uf.name] = str(e)
                        st.error(f"❌ {truncate(uf.name, 20)}: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
 
    # ── Paper list ────────────────────────────────────────────────────────────
    if st.session_state.papers:
        st.divider()
        st.caption("YOUR PAPERS")
        for pname, pdata in st.session_state.papers.items():
            is_active = pname == st.session_state.active_paper
            label = ("▶ " if is_active else "   ") + truncate(pname)
            if st.button(label, key=f"btn_{pname}"):
                st.session_state.active_paper = pname
                st.rerun()
            if is_active:
                st.caption(f"  {pdata['pages']} pages · {pdata['chunks']} chunks · {pdata['model']}")
 
        st.divider()
        if st.button("🗑️ Clear all papers", use_container_width=True):
            st.session_state.papers = {}
            st.session_state.active_paper = None
            st.session_state.conversations = {}
            st.rerun()
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
active = st.session_state.active_paper
paper_data = st.session_state.papers.get(active) if active else None
conversation = st.session_state.conversations.get(active, []) if active else []
 
if not active or not paper_data:
    # ── Empty / welcome state ─────────────────────────────────────────────────
    st.markdown('<div class="app-title">🧠 PaperMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">AI-powered research paper assistant</div>',
                unsafe_allow_html=True)
    st.divider()
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📎 **Upload** a PDF in the sidebar to get started.")
    with col2:
        st.info("💬 **Ask** questions in plain English — no prompting expertise needed.")
    with col3:
        st.info("📚 **Multi-paper** — switch between papers without losing your conversation.")
 
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📄</div>
        <p>No paper selected. Upload a PDF from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
 
else:
    # ── Active paper view ─────────────────────────────────────────────────────
    col_title, col_actions = st.columns([4, 1])
    with col_title:
        st.markdown(f"### 📄 {active}")
        stats_html = (
            f'<div class="paper-stats">'
            f'<span class="stat-badge">📖 {paper_data["pages"]} pages</span>'
            f'<span class="stat-badge">🔢 {paper_data["chunks"]} chunks</span>'
            f'<span class="stat-badge">🤖 {paper_data["model"]}</span>'
            f'</div>'
        )
        st.markdown(stats_html, unsafe_allow_html=True)
 
    with col_actions:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.conversations[active] = []
            st.rerun()
 
    st.divider()
 
    # ── Conversation history ──────────────────────────────────────────────────
    if not conversation:
        st.caption("No messages yet. Ask your first question below.")
    else:
        for msg in conversation:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><div class="chat-label">You</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-ai"><div class="chat-label">AI · {fmt_time(msg.get("ts",""))}</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
 
    st.divider()
 
    # ── Query input ───────────────────────────────────────────────────────────
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_area(
            "Ask a question about the paper",
            placeholder="e.g. What is the main contribution of this paper?",
            height=80,
            label_visibility="collapsed",
        )
        col_btn, col_hint = st.columns([1, 4])
        with col_btn:
            submitted = st.form_submit_button("Ask ↩", use_container_width=True)
        with col_hint:
            st.caption("Press **Ask** or Ctrl+Enter to submit.")
 
    if submitted and query.strip():
        with st.spinner("Thinking…"):
            try:
                result = paper_data["qa_chain"]({"query": query.strip()})
                answer = result.get("result", "No answer returned.")
 
                # Append with timestamp
                ts = datetime.now().isoformat()
                conversation.append({"role": "user", "content": query.strip(), "ts": ts})
                conversation.append({"role": "ai", "content": answer, "ts": ts})
                st.session_state.conversations[active] = conversation
                st.rerun()
 
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                st.info("Try rephrasing your question or check your API key / model settings.")
 
    elif submitted and not query.strip():
        st.warning("Please enter a question before submitting.")
 
    # ── Source chunks (optional debug) ───────────────────────────────────────
    if conversation:
        with st.expander("🔍 Show last retrieved sources", expanded=False):
            st.caption("Re-run the last query to see source chunks here if your QA chain returns them.")

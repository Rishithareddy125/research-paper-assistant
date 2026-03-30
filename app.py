
Copy

import streamlit as st
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from rag_pipeline import load_and_chunk_pdf, build_faiss_index, build_qa_chain, save_index, load_index
 
load_dotenv()
 
# ── Secure API key: env var or Streamlit secrets ONLY ────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
 
st.set_page_config(
    page_title="PaperMind — Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Load CSS from file (avoids tokenizer crash from inline CSS strings) ───────
_css_path = os.path.join(os.path.dirname(__file__), "styles.css")
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
 
# ── Session State ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("papers", {}),
    ("active_paper", None),
    ("conversations", {}),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Paper<span>Mind</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AI Research Assistant</div>', unsafe_allow_html=True)
 
    if groq_api_key:
        st.markdown(
            '<div style="font-size:0.75rem;color:#4ade80;margin-bottom:0.8rem;">'
            "🔒 API key configured</div>",
            unsafe_allow_html=True,
        )
    else:
        st.error("⚠️ GROQ_API_KEY not found.\nSet it in `.env` or Streamlit Secrets.")
 
    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        label_visibility="collapsed",
    )
 
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider("Chunks k", 2, 10, 5)
    with c2:
        chunk_size = st.slider("Chunk sz", 200, 1000, 500, 50)
 
    st.markdown('<div class="section-title">Upload Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
 
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.papers:
                if not groq_api_key:
                    st.error("⚠️ API key not configured.")
                else:
                    with st.spinner(f"Loading {uf.name[:18]}..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uf.read())
                                tmp_path = tmp.name
                            chunks, num_pages = load_and_chunk_pdf(tmp_path, chunk_size, 50)
                            vs = build_faiss_index(chunks)
                            qa = build_qa_chain(vs, model, top_k)
                            st.session_state.papers[uf.name] = {
                                "vectorstore": vs,
                                "qa_chain": qa,
                                "chunks": len(chunks),
                                "pages": num_pages,
                            }
                            st.session_state.conversations[uf.name] = []
                            if not st.session_state.active_paper:
                                st.session_state.active_paper = uf.name
                            os.unlink(tmp_path)
                            st.success("✅ Ready!")
                        except Exception as e:
                            st.error(f"Error: {e}")
 
    if st.session_state.papers:
        st.markdown('<div class="section-title">Your Papers</div>', unsafe_allow_html=True)
        for pname in st.session_state.papers:
            is_active = pname == st.session_state.active_paper
            short = pname[:24] + "…" if len(pname) > 24 else pname
            label = f"{'▶ ' if is_active else '  '}{short}"
            if st.button(label, key=f"p_{pname}", use_container_width=True):
                st.session_state.active_paper = pname
                st.rerun()
 
    st.markdown('<div class="section-title">Index</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.active_paper:
                save_index(
                    st.session_state.papers[st.session_state.active_paper]["vectorstore"]
                )
                st.success("Saved!")
            else:
                st.warning("No active paper.")
    with sc2:
        if st.button("📂 Load", use_container_width=True):
            vs = load_index()
            if vs:
                name = "loaded_index.pdf"
                qa = build_qa_chain(vs, model, top_k)
                st.session_state.papers[name] = {
                    "vectorstore": vs,
                    "qa_chain": qa,
                    "chunks": "?",
                    "pages": "?",
                }
                st.session_state.conversations[name] = []
                st.session_state.active_paper = name
                st.success("Loaded!")
            else:
                st.warning("No saved index.")
 
    _active_sidebar = st.session_state.active_paper
    if _active_sidebar and st.session_state.conversations.get(_active_sidebar):
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.conversations[_active_sidebar] = []
            st.rerun()
 
# ── Main ──────────────────────────────────────────────────────────────────────
active = st.session_state.active_paper
paper_data = st.session_state.papers.get(active) if active else None
conversation = st.session_state.conversations.get(active, []) if active else []
 
if active and paper_data:
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">📄 {active}</div>
                <div class="topbar-sub">Ask anything about this paper</div>
            </div>
            <div>
                <span class="stat-pill">📑 <strong>{paper_data['pages']}</strong> pages</span>
                <span class="stat-pill">🧩 <strong>{paper_data['chunks']}</strong> chunks</span>
                <span class="stat-pill">💬 <strong>{len(conversation) // 2}</strong> Q&amp;As</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="topbar-title">🧠 PaperMind</div>
                <div class="topbar-sub">Upload a research paper to get started</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
if not active or not paper_data:
    st.markdown(
        """
        <div class="welcome-wrap">
            <div class="welcome-icon">🧠</div>
            <div class="welcome-title">Ask your <span>paper</span> anything</div>
            <div class="welcome-sub">
                Upload one or more research PDFs from the sidebar and have
                a real conversation with your documents.
            </div>
            <div style="margin-top:1.5rem;color:#3a3a5a;font-size:0.8rem;">
                ← Upload PDFs from the sidebar to begin
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
else:
    if not conversation:
        st.markdown(
            """
            <div class="welcome-wrap" style="min-height:40vh">
                <div class="welcome-icon">💡</div>
                <div class="welcome-title" style="font-size:1.2rem">Paper ready! Start asking.</div>
                <div class="welcome-sub">Pick a suggestion or type your own question below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
        suggestions = [
            "What is the main contribution?",
            "What methodology was used?",
            "What are the key results?",
            "What datasets were used?",
            "What are the limitations?",
            "How does this compare to prior work?",
        ]
        cols = st.columns(3)
        for i, sug in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            result = paper_data["qa_chain"]({"query": sug})
                            answer = result["result"]
                            sources = list(
                                set(
                                    doc.metadata.get("page", "?")
                                    for doc in result.get("source_documents", [])
                                )
                            )
                            now = datetime.now().strftime("%H:%M")
                            conversation.append({"role": "user", "content": sug, "time": now})
                            conversation.append(
                                {"role": "ai", "content": answer, "sources": sources, "time": now}
                            )
                            st.session_state.conversations[active] = conversation
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        for msg in conversation:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="msg-row user">
                        <div class="avatar uav">👤</div>
                        <div>
                            <div class="bubble usr">{msg['content']}</div>
                            <div class="btime">{msg.get('time', '')}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                sources_html = ""
                if msg.get("sources"):
                    tags = "".join(
                        f'<span class="src-tag">📖 p.{p}</span>'
                        for p in sorted(set(msg["sources"]))
                    )
                    sources_html = f'<div class="source-row">{tags}</div>'
                st.markdown(
                    f"""
                    <div class="msg-row">
                        <div class="avatar ai">🧠</div>
                        <div>
                            <div class="bubble ai">{msg['content']}{sources_html}</div>
                            <div class="btime">{msg.get('time', '')}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
 
    st.markdown("---")
    col_q, col_btn = st.columns([11, 1])
    with col_q:
        query = st.text_input(
            "Ask",
            placeholder="Ask anything about the paper...",
            label_visibility="collapsed",
            key="q_input",
        )
    with col_btn:
        send = st.button("➤", use_container_width=True)
 
    if send and query and query.strip():
        with st.spinner(" Thinking..."):
            try:
                history_context = ""
                if len(conversation) >= 2:
                    recent = conversation[-6:]
                    history_context = "\n\nPrevious conversation for context:\n"
                    for m in recent:
                        role = "User" if m["role"] == "user" else "Assistant"
                        history_context += f"{role}: {m['content']}\n"
 
                full_query = query + history_context if history_context else query
                result = paper_data["qa_chain"]({"query": full_query})
                answer = result["result"]
                sources = list(
                    set(
                        doc.metadata.get("page", "?")
                        for doc in result.get("source_documents", [])
                    )
                )
                now = datetime.now().strftime("%H:%M")
                conversation.append({"role": "user", "content": query, "time": now})
                conversation.append(
                    {"role": "ai", "content": answer, "sources": sources, "time": now}
                )
                st.session_state.conversations[active] = conversation
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

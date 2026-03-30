import streamlit as st
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from rag_pipeline import load_and_chunk_pdf, build_faiss_index, build_qa_chain, save_index, load_index
 
load_dotenv()
 
st.set_page_config(
    page_title="PaperMind — Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
 
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a0f;
    color: #e8e8f0;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3f; border-radius: 2px; }
 
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
 
/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d15 !important;
    border-right: 1px solid #1a1a2e !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem; }
 
.brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    letter-spacing: -0.5px; margin-bottom: 0.2rem;
}
.brand span { color: #7c6aff; }
.brand-sub {
    font-size: 0.72rem; color: #4a4a6a;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.5rem;
}
.section-title {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #3a3a5a;
    font-weight: 600; margin: 1.2rem 0 0.6rem;
}
 
[data-testid="stSidebar"] label {
    color: #6a6a8a !important; font-size: 0.72rem !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] input {
    background: #13131f !important; border: 1px solid #1e1e30 !important;
    color: #e8e8f0 !important; border-radius: 8px !important; font-size: 0.85rem !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #13131f !important; border: 1px solid #1e1e30 !important;
    border-radius: 8px !important; color: #e8e8f0 !important;
}
[data-testid="stSidebar"] button {
    background: #13131f !important; border: 1px solid #1e1e30 !important;
    color: #9a9abf !important; border-radius: 8px !important; font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] button:hover {
    border-color: #7c6aff !important; color: #e8e8f0 !important; background: #1a1a2e !important;
}
 
/* Topbar */
.topbar {
    padding: 1rem 2rem; border-bottom: 1px solid #1a1a2e;
    display: flex; align-items: center; justify-content: space-between;
    background: #0a0a0f;
}
.topbar-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
.topbar-sub { font-size: 0.75rem; color: #4a4a6a; margin-top: 1px; }
.stat-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #13131f; border: 1px solid #1e1e30;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #6a6a8a; margin-left: 8px;
}
.stat-pill strong { color: #7c6aff; font-weight: 600; }
 
/* Welcome */
.welcome-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 55vh;
    text-align: center; gap: 0.8rem; padding: 2rem;
}
.welcome-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.welcome-title {
    font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; letter-spacing: -0.5px;
}
.welcome-title span { color: #7c6aff; }
.welcome-sub { font-size: 0.9rem; color: #4a4a6a; max-width: 400px; line-height: 1.6; }
 
/* Chat messages */
.msg-row { display: flex; gap: 12px; margin-bottom: 1.2rem; animation: fadeUp 0.3s ease; }
.msg-row.user { flex-direction: row-reverse; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.avatar {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0; margin-top: 2px;
}
.avatar.ai { background: #1a1a2e; border: 1px solid #2a2a4a; }
.avatar.uav { background: #7c6aff20; border: 1px solid #7c6aff40; }
 
.bubble {
    max-width: 72%; border-radius: 14px;
    padding: 12px 16px; font-size: 0.88rem; line-height: 1.65;
}
.bubble.ai {
    background: #13131f; border: 1px solid #1e1e30;
    color: #d8d8f0; border-top-left-radius: 4px;
}
.bubble.usr {
    background: #7c6aff; color: #ffffff; border-top-right-radius: 4px;
}
.btime { font-size: 0.68rem; color: #3a3a5a; margin-top: 4px; }
 
.source-row { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.src-tag {
    background: #1e1e30; color: #7c6aff;
    border-radius: 6px; padding: 2px 8px;
    font-size: 0.7rem; font-weight: 500;
}
 
/* Input */
[data-testid="stTextInput"] input {
    background: #13131f !important; border: 1px solid #2a2a3f !important;
    color: #e8e8f0 !important; border-radius: 10px !important;
    font-size: 0.9rem !important; padding: 10px 14px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c6aff !important;
    box-shadow: 0 0 0 3px #7c6aff15 !important;
}
[data-testid="stTextInput"] input::placeholder { color: #3a3a5a !important; }
 
/* Main buttons */
.stButton > button {
    background: #7c6aff !important; border: none !important;
    color: white !important; border-radius: 10px !important;
    font-weight: 600 !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
 
/* File uploader */
[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 1px dashed #2a2a3f !important;
    border-radius: 12px !important;
}
 
hr { border-color: #1a1a2e !important; }
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)
 
# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("papers", {}),
    ("active_paper", None),
    ("conversations", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Paper<span>Mind</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AI Research Assistant</div>', unsafe_allow_html=True)
 
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.getenv("GROQ_API_KEY", ""), placeholder="gsk_...")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
 
    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    model = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ], label_visibility="collapsed")
 
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider("Chunks k", 2, 10, 5)
    with c2:
        chunk_size = st.slider("Chunk sz", 200, 1000, 500, 50)
 
    st.markdown('<div class="section-title">Upload Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDFs", type=["pdf"],
                                       accept_multiple_files=True,
                                       label_visibility="collapsed")
 
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.papers:
                if not os.getenv("GROQ_API_KEY"):
                    st.error("⚠️ Enter API key first.")
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
                                "vectorstore": vs, "qa_chain": qa,
                                "chunks": len(chunks), "pages": num_pages,
                            }
                            st.session_state.conversations[uf.name] = []
                            if not st.session_state.active_paper:
                                st.session_state.active_paper = uf.name
                            os.unlink(tmp_path)
                            st.success(f"✅ Ready!")
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
                save_index(st.session_state.papers[st.session_state.active_paper]["vectorstore"])
                st.success("Saved!")
            else:
                st.warning("No active paper.")
    with sc2:
        if st.button("📂 Load", use_container_width=True):
            vs = load_index()
            if vs:
                name = "loaded_index.pdf"
                qa = build_qa_chain(vs, model, top_k)
                st.session_state.papers[name] = {"vectorstore": vs, "qa_chain": qa, "chunks": "?", "pages": "?"}
                st.session_state.conversations[name] = []
                st.session_state.active_paper = name
                st.success("Loaded!")
            else:
                st.warning("No saved index.")
 
    active = st.session_state.active_paper
    if active and st.session_state.conversations.get(active):
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.conversations[active] = []
            st.rerun()
 
# ── Main ──────────────────────────────────────────────────────────────────────
active = st.session_state.active_paper
paper_data = st.session_state.papers.get(active) if active else None
conversation = st.session_state.conversations.get(active, []) if active else []
 
# Topbar
if active and paper_data:
    st.markdown(f"""
    <div class="topbar">
        <div>
            <div class="topbar-title">📄 {active}</div>
            <div class="topbar-sub">Ask anything about this paper</div>
        </div>
        <div>
            <span class="stat-pill">📑 <strong>{paper_data['pages']}</strong> pages</span>
            <span class="stat-pill">🧩 <strong>{paper_data['chunks']}</strong> chunks</span>
            <span class="stat-pill">💬 <strong>{len(conversation)//2}</strong> Q&amp;As</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="topbar">
        <div>
            <div class="topbar-title">🧠 PaperMind</div>
            <div class="topbar-sub">Upload a research paper to get started</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
# No paper uploaded
if not active or not paper_data:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🧠</div>
        <div class="welcome-title">Ask your <span>paper</span> anything</div>
        <div class="welcome-sub">Upload one or more research PDFs from the sidebar and have a real conversation with your documents.</div>
        <div style="margin-top:1.5rem; color:#3a3a5a; font-size:0.8rem;">← Upload PDFs from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
 
else:
    # Suggestion buttons when no conversation yet
    if not conversation:
        st.markdown("""
        <div class="welcome-wrap" style="min-height:40vh">
            <div class="welcome-icon">💡</div>
            <div class="welcome-title" style="font-size:1.2rem">Paper ready! Start asking.</div>
            <div class="welcome-sub">Pick a suggestion or type your own question below.</div>
        </div>
        """, unsafe_allow_html=True)
 
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
                            sources = list(set([doc.metadata.get("page", "?") for doc in result.get("source_documents", [])]))
                            now = datetime.now().strftime("%H:%M")
                            conversation.append({"role": "user", "content": sug, "time": now})
                            conversation.append({"role": "ai", "content": answer, "sources": sources, "time": now})
                            st.session_state.conversations[active] = conversation
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
 
    else:
        # Render chat messages
        for msg in conversation:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="avatar uav">👤</div>
                    <div>
                        <div class="bubble usr">{msg['content']}</div>
                        <div class="btime">{msg.get('time','')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    tags = "".join([f'<span class="src-tag">📖 p.{p}</span>' for p in sorted(set(msg["sources"]))])
                    sources_html = f'<div class="source-row">{tags}</div>'
                st.markdown(f"""
                <div class="msg-row">
                    <div class="avatar ai">🧠</div>
                    <div>
                        <div class="bubble ai">{msg['content']}{sources_html}</div>
                        <div class="btime">{msg.get('time','')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
    # Input
    st.markdown("---")
    col_q, col_btn = st.columns([11, 1])
    with col_q:
        query = st.text_input("Ask", placeholder="Ask anything about the paper...",
                              label_visibility="collapsed", key="q_input")
    with col_btn:
        send = st.button("➤", use_container_width=True)
 
    if send and query and query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                # Conversation memory — include last 3 exchanges
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
                sources = list(set([doc.metadata.get("page", "?") for doc in result.get("source_documents", [])]))
                now = datetime.now().strftime("%H:%M")
                conversation.append({"role": "user", "content": query, "time": now})
                conversation.append({"role": "ai", "content": answer, "sources": sources, "time": now})
                st.session_state.conversations[active] = conversation
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

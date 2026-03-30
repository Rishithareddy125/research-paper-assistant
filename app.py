import streamlit as st
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from rag_pipeline import load_and_chunk_pdf, build_faiss_index, build_qa_chain, save_index, load_index

load_dotenv()

# ✅ Secure API key
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

st.set_page_config(
    page_title="PaperMind — Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.markdown("### 🧠 PaperMind")
    st.markdown("AI Research Assistant")

    # ❌ Removed API input field

    if not groq_api_key:
        st.error("⚠️ API key not found. Set it in Streamlit Secrets.")

    model = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    top_k = st.slider("Chunks k", 2, 10, 5)
    chunk_size = st.slider("Chunk size", 200, 1000, 500, 50)

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.papers:
                if not groq_api_key:
                    st.error("⚠️ API key not configured.")
                else:
                    with st.spinner(f"Processing {uf.name}..."):
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
        st.markdown("### Your Papers")
        for pname in st.session_state.papers:
            if st.button(pname):
                st.session_state.active_paper = pname
                st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
active = st.session_state.active_paper
paper_data = st.session_state.papers.get(active) if active else None
conversation = st.session_state.conversations.get(active, []) if active else []

if not active or not paper_data:
    st.title("🧠 PaperMind")
    st.write("Upload a research paper and start asking questions.")
else:
    st.title(f"📄 {active}")

    for msg in conversation:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")

    query = st.text_input("Ask a question about the paper:")

    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            try:
                result = paper_data["qa_chain"]({"query": query})
                answer = result["result"]

                conversation.append({"role": "user", "content": query})
                conversation.append({"role": "ai", "content": answer})

                st.session_state.conversations[active] = conversation
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

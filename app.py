import streamlit as st
from rag_pipeline import ingest_uploaded_files, load_llm, get_answer

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Chat with CVs",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0e1117; color: #e6edf3; }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .streamlit-expanderHeader {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        color: #58a6ff !important;
    }
    .chunk-text {
        background-color: #0d1117;
        border-left: 3px solid #58a6ff;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 0 6px 6px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #8b949e;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .badge {
        display: inline-block;
        background-color: #1f6feb;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
        margin-right: 4px;
    }
    .badge-page {
        background-color: transparent;
        color: #58a6ff;
        border: 1px solid #388bfd;
    }
    hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)


# ── Load LLM once ─────────────────────────────────────────
@st.cache_resource
def get_llm():
    return load_llm()

llm = get_llm()


# ── Sidebar — File Uploader ───────────────────────────────
with st.sidebar:
    st.markdown("## 💼 CV Chat")
    st.markdown("*HR Assistant powered by RAG*")
    st.divider()

    st.markdown("### 📎 Upload CVs")
    uploaded_files = st.file_uploader(
        label="Upload PDF CVs (up to 5)",
        type=["pdf"],
        accept_multiple_files=True,
        key="cv_uploader"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready")
        for f in uploaded_files:
            st.markdown(f"&nbsp;&nbsp;📄 `{f.name}`", unsafe_allow_html=True)

        # Only re-ingest if the uploaded files actually changed
        current_file_names = sorted([f.name for f in uploaded_files])
        if st.session_state.get("loaded_files") != current_file_names:
            with st.spinner("⚙️ Reading and indexing CVs..."):
                st.session_state.vectorstore  = ingest_uploaded_files(uploaded_files)
                st.session_state.loaded_files = current_file_names
                st.session_state.messages     = []
            st.success("🚀 Ready! Ask your questions.")
    else:
        st.info("👆 Upload your CVs to get started.")

    st.divider()

    # Sample questions
    st.markdown("### 💡 Try asking")
    samples = [
        "Who knows Python?",
        "Who has the most experience?",
        "Who studied Computer Science?",
        "Best candidate for a data role?",
        "Who worked at a tech company?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Helper: render source chunks ──────────────────────────
def render_chunks(chunks):
    st.divider()
    st.markdown("##### 📄 Retrieved Source Chunks")
    for i, chunk in enumerate(chunks):
        cv_name  = chunk.metadata.get("cv_name", "Unknown")
        page_num = int(chunk.metadata.get("page", 0)) + 1
        with st.expander(f"Chunk {i+1}  ·  {cv_name}  ·  Page {page_num}"):
            st.markdown(
                f'<span class="badge">{cv_name}</span>'
                f'<span class="badge badge-page">Page {page_num}</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chunk-text">{chunk.page_content}</div>',
                unsafe_allow_html=True
            )


# ── Main area ─────────────────────────────────────────────
st.markdown("## 💼 Chat with CVs")
st.markdown("Ask anything about the uploaded candidates — skills, experience, education, or role fit.")
st.divider()

# Guard: don't show chat until CVs are uploaded
if "vectorstore" not in st.session_state:
    st.markdown("### 👈 Upload your CVs from the sidebar to begin")
    st.stop()

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            render_chunks(msg["chunks"])

# Prefill from sidebar button clicks
prefill = st.session_state.pop("prefill", None)

# Chat input
query = st.chat_input("Ask about the candidates...") or prefill

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching CVs..."):
            try:
                answer, chunks = get_answer(query, st.session_state.vectorstore, llm)
            except Exception as e:
                answer = f"⚠️ Something went wrong: {str(e)}"
                chunks = []

        st.markdown(answer)
        if chunks:
            render_chunks(chunks)

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "chunks" : chunks
    })
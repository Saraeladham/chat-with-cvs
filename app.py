"""
app.py
------
Clean, minimal Streamlit UI for Chat with CVs.

Layout:
    - No sidebar
    - Title + subtitle
    - CV uploader
    - Chunking strategy selector
    - Chat interface
    - Ranked answer + collapsible source chunks

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag_pipeline import ingest_uploaded_files, load_llm, get_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Assistant",
    page_icon="💼",
    layout="centered",
)

# ── CSS — single dark background, clean minimal style ─────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Single background color across everything */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], section.main, .block-container {
        background-color: #111318 !important;
        color: #e2e8f0;
        font-family: 'Sora', sans-serif;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    /* Title */
    .app-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }

    .app-sub {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 28px;
        font-weight: 300;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #1e2330;
        margin: 20px 0;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1d27 !important;
        border: 1px dashed #2e3347 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    /* Select box */
    [data-testid="stSelectbox"] > div > div {
        background-color: #1a1d27 !important;
        border: 1px solid #2e3347 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-size: 0.85rem !important;
    }

    /* Status messages */
    .status-box {
        background-color: #1a1d27;
        border: 1px solid #2e3347;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .status-ready {
        border-left: 3px solid #22c55e;
        color: #86efac;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1a1d27 !important;
        border: 1px solid #1e2330 !important;
        border-radius: 10px !important;
        padding: 14px !important;
        margin-bottom: 10px !important;
    }

    /* Answer text */
    .answer-text {
        font-size: 0.93rem;
        line-height: 1.75;
        color: #e2e8f0;
        white-space: pre-wrap;
    }

    /* Source chunk expander */
    .streamlit-expanderHeader {
        background-color: #1a1d27 !important;
        border: 1px solid #2e3347 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        color: #475569 !important;
    }

    .streamlit-expanderContent {
        background-color: #13151e !important;
        border: 1px solid #2e3347 !important;
        border-top: none !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.77rem;
        color: #64748b;
        line-height: 1.7;
    }

    /* Metadata tags */
    .meta-tag {
        display: inline-block;
        background-color: #1e2330;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 4px;
        margin-bottom: 6px;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: #1a1d27 !important;
        border: 1px solid #2e3347 !important;
        border-radius: 10px !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load LLM once ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Caches the LLM instance across Streamlit reruns.
    Without caching, a new OpenAI client would be created on every interaction.
    """
    return load_llm()


llm = get_llm()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">💼 HR Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Chat with CVs — document-aware candidate ranking</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ── Upload Section ────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    label="Upload CV files (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    key="cv_uploader",
    label_visibility="collapsed",
    help="Upload up to 5 candidate CVs in PDF format"
)

# Chunking strategy selector — shown only after files are uploaded
chunk_strategy = "recursive"   # default
if uploaded_files:
    chunk_strategy = st.selectbox(
        "Chunking strategy",
        options=["recursive", "semantic"],
        format_func=lambda x: "Recursive (character-based, fast)" if x == "recursive" else "Semantic (meaning-based, accurate)",
        key="chunk_strategy",
        label_visibility="visible",
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ── Ingestion Logic ───────────────────────────────────────────────────────────
if uploaded_files:
    # Adaptive key: only re-ingest when files or strategy actually change
    # Value: sorted list of filenames + strategy string
    # This prevents rebuilding the vectorstore on every Streamlit rerun
    current_state = sorted([f.name for f in uploaded_files]) + [chunk_strategy]

    if st.session_state.get("loaded_state") != current_state:
        with st.spinner(f"⚙️ Reading {len(uploaded_files)} CV(s) and indexing..."):
            try:
                st.session_state.vectorstore  = ingest_uploaded_files(
                    uploaded_files, chunk_strategy, llm
                )
                st.session_state.loaded_state = current_state
                st.session_state.messages     = []   # reset chat on new upload
            except Exception as e:
                st.error(f"❌ Ingestion failed: {str(e)}")
                st.stop()

    # Show ready status with file list
    file_names = [f.name for f in uploaded_files]
    files_str  = "  ·  ".join([f"📄 {n}" for n in file_names])
    st.markdown(
        f'<div class="status-box status-ready">✓ Ready  —  {files_str}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="status-box">Upload CV files above to begin</div>',
        unsafe_allow_html=True
    )


# ── Source chunks renderer ────────────────────────────────────────────────────
def render_chunks(chunks: list):
    """
    Renders retrieved source chunks as collapsible expanders below the answer.

    Each expander shows:
        - Candidate name, CV filename, page number, total pages, chunk index
        - Raw chunk text in monospace font

    This gives the user full transparency into WHAT the LLM read
    to generate the ranked answer.

    Args:
        chunks (list): Retrieved LangChain Document objects with metadata
    """
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(f"📎 Source chunks ({len(chunks)} retrieved)", expanded=False):
        for i, chunk in enumerate(chunks):
            m            = chunk.metadata
            candidate    = m.get("candidate_name", "Unknown")
            cv_name      = m.get("cv_name", "?")
            page_num     = m.get("page_number", "?")
            total_pages  = m.get("total_pages", "?")
            chunk_idx    = m.get("chunk_index", "?")

            st.markdown(
                f'<span class="meta-tag">{candidate}</span>'
                f'<span class="meta-tag">{cv_name}</span>'
                f'<span class="meta-tag">p.{page_num}/{total_pages}</span>'
                f'<span class="meta-tag">chunk #{chunk_idx}</span>',
                unsafe_allow_html=True
            )
            st.caption(chunk.page_content)
            if i < len(chunks) - 1:
                st.markdown("---")


# ── Chat Interface ────────────────────────────────────────────────────────────

# Guard: don't show chat until CVs are uploaded and indexed
if "vectorstore" not in st.session_state:
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="answer-text">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("chunks"):
            render_chunks(msg["chunks"])

# Chat input
query = st.chat_input("Ask about the candidates...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing CVs..."):
            try:
                answer, chunks = get_answer(query, st.session_state.vectorstore, llm)
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
                chunks = []

        st.markdown(f'<div class="answer-text">{answer}</div>', unsafe_allow_html=True)
        if chunks:
            render_chunks(chunks)

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "chunks" : chunks
    })
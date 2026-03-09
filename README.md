# 💼 Chat with CVs — Document-Aware RAG HR Assistant
A production-grade HR assistant that lets you upload candidate CVs and ask natural language questions. Built on a Document-Aware RAG pipeline that ranks candidates by depth of evidence, not just keyword presence.

# 🚀 What's New (Latest Updates)
# 1. Naive RAG → Document-Aware RAG

Before: Retrieved chunks were stuffed into the prompt with no structure — the LLM had no idea which chunk belonged to which candidate.
After: Chunks are grouped by candidate in the prompt. The LLM reasons per-candidate and returns a ranked list based on depth of evidence (projects > skills section mention).

# 2. Fixed Multi-Page CV Reading

Before: PyPDFLoader only reliably read page 1 of each CV — information on later pages was silently missed.
After: Switched to PDFPlumberLoader which reads ALL pages including complex layouts and formatted text.

# 3. Rich Metadata Tagging

Before: Each chunk only knew the filename.
After: Every chunk is tagged with:

candidate_name — extracted from CV by LLM
page_number — which page the chunk came from (1-indexed)
total_pages — total pages in the CV
cv_name — original filename
chunk_index — position within the document



# 4. Candidate Name Extraction

LLM reads the first page of each CV and extracts the candidate's real name automatically.
Answers now reference candidates by name, not by filename.

# 5. Two Chunking Strategies (User Selectable)

Recursive (default): RecursiveCharacterTextSplitter — 500 chars, 50 overlap. Fast and predictable.
Semantic: SemanticChunker — splits by meaning, keeps related sections intact (e.g. full Skills section stays together). Better for varied CV formats, slower.

# 6. Prompt Injection Guardrails

The prompt is structured so instructions appear after the user query — LLM pays more attention to instructions closer to generation.
Explicit rules added to reject off-topic questions, joke requests, and instruction override attempts.
Off-topic queries return: "I can only answer questions about the uploaded CVs."

# 7. Embedding Model Fix

Updated import from deprecated langchain_community.embeddings to langchain_huggingface.
Added global _embeddings_cache to load the model once per session — prevents reloading on every CV file processed.

# 8. LLM Flexibility

Switched between OpenAI → Google Gemini → Groq across iterations.
Currently running on Groq llama-3.3-70b-versatile — free, fast, no daily quota limits.
Easy to swap: only load_llm() needs to change.

# 9. Adaptive Re-Ingestion Key

The vectorstore only rebuilds when uploaded files or chunking strategy actually change.
Prevents redundant re-embedding on every Streamlit UI interaction.


# 🧠 Pipeline
OFFLINE (per upload)
─────────────────────────────────────────────────────
PDF Upload (Streamlit)
    → PDFPlumberLoader (all pages)
    → LLM extracts candidate name from page 1
    → Rich metadata tagging (name, page, total pages)
    → Chunking: Recursive (500/50) or Semantic
    → HuggingFace all-MiniLM-L6-v2 → 384-dim vectors
    → Qdrant in-memory vector store

ONLINE (per query)
─────────────────────────────────────────────────────
User Question
    → Prompt injection guard
    → Qdrant cosine similarity → Top-10 chunks
    → Chunks grouped by candidate_name
    → Document-aware ranked prompt
    → Groq LLaMA 3.3 70B → Ranked answer
    → Streamlit: answer + collapsible source chunks

# 🛠️ Tech Stack
PDF Loading ----> PDFPlumberLoader (LangChain)
Text Splitting ----> RecursiveCharacterTextSplitter / SemanticChunker
Embeddings ----> all-MiniLM-L6-v2 (HuggingFace)
Vector Store ----> Qdrant (in-memory)
LLM ---->Groq — LLaMA 3.3 70B Versatile
Framework ----> LangChain
UI ----> Streamlit

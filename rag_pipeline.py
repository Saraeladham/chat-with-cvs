import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

load_dotenv()

COLLECTION_NAME = "cvs_collection"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 3


def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    return ChatGroq(
        api_key=api_key,
        # model_name="llama3-8b-8192"
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
    )


def ingest_uploaded_files(uploaded_files: list) -> QdrantVectorStore:
    docs = []

    for uploaded_file in uploaded_files:
        # Streamlit gives us a file object, not a path
        # PyPDFLoader needs a real file path, so we save it temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()

        for page in pages:
            page.metadata["cv_name"] = uploaded_file.name

        docs.extend(pages)
        os.remove(tmp_path)  # delete temp file after loading

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Embed
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # Build Qdrant in memory — no folder, no server needed
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def get_answer(query: str, vectorstore: QdrantVectorStore, llm: ChatGroq) -> tuple:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    chunks = retriever.invoke(query)

    context_parts = []
    for chunk in chunks:
        cv_name  = chunk.metadata.get("cv_name", "Unknown")
        page_num = chunk.metadata.get("page", "?")
        context_parts.append(
            f"--- [{cv_name} | Page {page_num}] ---\n{chunk.page_content}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a professional HR assistant.
Answer the question based ONLY on the CV content below.
Be specific. Mention candidate names when relevant.
If the answer is not in the CVs, say "I couldn't find that information."
Do NOT make anything up.

CV Content:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, chunks
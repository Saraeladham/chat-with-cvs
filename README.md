# 💼 Chat with CVs

An HR assistant that lets you ask natural language questions across uploaded CVs using a full RAG pipeline.

## Stack
- LangChain + Qdrant (in-memory) + Groq (LLaMA 3.3) + Streamlit

## How to Run
1. Clone the repo
2. Create a virtual environment and activate it
3. Run `pip install -r requirements.txt`
4. Create a `.env` file and add your `GROQ_API_KEY`
5. Run `streamlit run app.py`
6. Upload your CVs from the sidebar and start chatting
# RAG Chat Application

This project implements a Retrieval-Augmented Generation (RAG) pipeline using:

- LangChain
- FAISS
- HuggingFace Embeddings
- Qwen LLM
- Streamlit UI
- ReportLab (PDF export)

## Features

- Retrieval-Augmented Generation (RAG) using FAISS
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Qwen 0.5B Instruct LLM (CPU-based)
- Streamlit Chat UI (ChatGPT-style)
- Chat history with session state
- PDF Export (Direct Method)
- PDF Export via MCP Tool Calling
- Prompt engineering with strict context enforcement


🔹 MCP Integration

This project demonstrates Model Context Protocol (MCP)-style tool integration where the LLM can trigger structured tools (PDF generator) instead of handling file creation directly.


## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run Streamlit:
   streamlit run app.py

# RAG Prototype – University Scholarship Documentation

This project implements a local Retrieval-Augmented Generation (RAG) system designed to answer questions about a university scholarship regulation document.  
The pipeline is fully local and built using LangChain, Ollama, and ChromaDB.

## Features

- **PDF ingestion** with automatic text extraction  
- **Chunking** using `RecursiveCharacterTextSplitter`  
- **Local embeddings** via `nomic-embed-text` (Ollama)  
- **Vector database** with persistent ChromaDB storage  
- **Hybrid retrieval** combining semantic search and BM25  
- **Reciprocal Rank Fusion (RRF)** for merging results  
- **CrossEncoder reranking** for improved precision  
- **Query expansion** using an LLM to increase recall  
- **Final answer generation** using a local LLM (Ollama)  
- **CLI interface** for interactive question answering

## Usage

1. Place your PDF document(s) inside the `data/` folder.  
2. Run `ingest.py` to build the vector database.  
3. Run `chat.py` to start the interactive CLI chatbot.  

The system will answer questions using only the information contained in the indexed documents.

## Technologies

- LangChain  
- Ollama (local LLMs + embeddings)  
- ChromaDB  
- BM25  
- CrossEncoder (ms‑marco‑MiniLM‑L‑6‑v2)  
- Python

# ⚖️ AI-Powered Legal Assistant – Hybrid RAG Approach

This project is an intelligent legal chatbot built using **Hybrid Search (dense + sparse retrieval)**, combining top transformer models and metadata filtering for precise document answers.

## 🚀 Features

- 🔍 Hybrid Search: Combines vector (dense) + keyword (sparse) search
- ⚡ Metadata filtering for faster and more relevant matches
- 🧠 Reranking using `all-mpnet-base-v2` for accurate answers
- 💬 Streamlit UI (for demo), customizable for production
- 🗂️ Supports Pinecone, Chroma, FAISS, and MongoDB Atlas

## 🛠️ Tech Stack

- Python, Streamlit
- Azure OpenAI (GPT-4o, embeddings)
- Sentence Transformers
- Pinecone (vector DB)
- dotenv, re, json, os

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

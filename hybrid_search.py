import os
import time
import re
import json
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv(".env")

# Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Pinecone setup
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_index = pinecone.Index(index_name)

# SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Hybrid Search Logic ---
def embed_query(query):
    return embedding_model.encode(query).tolist()

def search_pinecone_dense(query_vector, top_k=10):
    return pinecone_index.query(vector=query_vector, top_k=top_k, include_metadata=True)["matches"]

def search_bm25(query, documents, top_k=10):
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [documents[i] for i in ranked_indices]

def rerank_chunks(query, texts):
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embs = embedding_model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, chunk_embs)[0]
    best_idx = int(scores.argmax())
    return texts[best_idx], float(scores[best_idx])

def generate_answer(question, context):
    prompt = f"""You are a legal assistant AI. Use the following metadata and summary to answer the question accurately.

Context:
{context}

Question: {question}
Answer:"""
    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def hybrid_search(query):
    # Step 1: Dense retrieval
    query_vector = embed_query(query)
    matches = search_pinecone_dense(query_vector)

    if not matches:
        return "No matches found", 0.0

    # Step 2: Prepare enriched chunks
    enriched_chunks = []
    raw_texts = []
    for m in matches:
        meta = m["metadata"]
        summary = meta.get("summary", "")
        keywords = ", ".join(meta.get("keywords", []))
        entities = ", ".join(meta.get("named_entities", []))
        intent = meta.get("intent", "")
        enriched = f"Summary: {summary}\nKeywords: {keywords}\nEntities: {entities}\nIntent: {intent}"
        enriched_chunks.append(enriched)
        raw_texts.append(summary)

    # Step 3: Combine with BM25 filtering
    bm25_filtered = search_bm25(query, raw_texts, top_k=5)
    filtered_enriched_chunks = [
        chunk for chunk in enriched_chunks if any(b in chunk for b in bm25_filtered)
    ]

    if not filtered_enriched_chunks:
        filtered_enriched_chunks = enriched_chunks[:5]  # fallback

    # Step 4: Rerank
    best_context, score = rerank_chunks(query, filtered_enriched_chunks)
    return best_context, score

# --- Main ---
def main():
    print("❓ Enter your question (or 'exit'):")

    while True:
        query = input("❓ ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        start_time = time.time()
        print("🔍 Running hybrid search...")

        best_context, score = hybrid_search(query)
        answer = generate_answer(query, best_context)
        end_time = time.time()

        print(f"\n💬 Answer: {answer}")
        print(f"📊 Rerank Score: {score:.2f}")
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds\n")

if __name__ == "__main__":
    main()

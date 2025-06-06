import os
import time
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv(".env")

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_index = pinecone.Index(index_name)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_query(query):
    return embedding_model.encode(query).tolist()

def search_pinecone(query_vector, top_k=5):
    response = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response["matches"]

def rerank_chunks(query, matches):
    chunks = []
    for m in matches:
        meta = m["metadata"]
        summary = meta.get("summary", "")
        keywords = ", ".join(meta.get("keywords", []))
        entities = ", ".join(meta.get("named_entities", []))
        intent = meta.get("intent", "")
        raw_text = meta.get("text", "")
        enriched = (
            f"Summary: {summary}\n"
            f"Keywords: {keywords}\n"
            f"Entities: {entities}\n"
            f"Intent: {intent}\n"
            f"Raw Text: {raw_text}"
        )
        chunks.append(enriched)

    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embs = embedding_model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, chunk_embs)[0]

    best_idx = int(scores.argmax())
    return chunks[best_idx], float(scores[best_idx])

def generate_answer(question, context):
    prompt = f"""You are a legal assistant AI. Use the following metadata, summary, and raw text to answer the question accurately.

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

def main():
    print("❓ Enter your question (or 'exit'):")

    while True:
        query = input("❓ ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        start_time = time.time()
        print("🔍 Searching in Pinecone...")

        query_vector = embed_query(query)
        matches = search_pinecone(query_vector)

        if not matches:
            print("⚠️ No relevant results found.")
            continue

        best_context, score = rerank_chunks(query, matches)
        answer = generate_answer(query, best_context)

        end_time = time.time()

        print(f"\n💬 Answer: {answer}")
        print(f"📊 Rerank Score: {score:.2f}")
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds\n")

if __name__ == "__main__":
    main()

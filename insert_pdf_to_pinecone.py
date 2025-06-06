import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from keybert import KeyBERT
from transformers import pipeline

# ====== Load .env credentials ======
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
namespace = "__default__"

# ====== Load Models ======
embedding_model = SentenceTransformer("all-mpnet-base-v2")
kw_model = KeyBERT()
ner_model = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ====== Hardcoded PDF Path ======
pdf_path = r"C:\Users\naape\Downloads\Rag_dense_project\data\MHC_CaseStatus_511655.pdf"

# ====== Step 1: Chunk PDF ======
def read_pdf_chunks(pdf_path, max_chunk_len=500):
    reader = PdfReader(pdf_path)
    full_text = "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )
    words = full_text.split()
    chunks = [
        " ".join(words[i:i + max_chunk_len]) for i in range(0, len(words), max_chunk_len)
    ]
    return chunks, full_text

# ====== Step 2: Metadata Extraction ======
def extract_metadata(full_text):
    keywords = [kw[0] for kw in kw_model.extract_keywords(full_text, top_n=10)]
    intent = "judgement_summary" if "judgement" in full_text.lower() else "legal_case"
    entities = ner_model(full_text[:4000])
    named_entities = [f"{e['word']} ({e['entity_group']})" for e in entities]
    summary = summarizer(full_text[:1024])[0]["summary_text"]

    return {
        "intent": intent,
        "keywords": keywords,
        "named_entities": named_entities,
        "summary": summary
    }

# ====== Step 3: Embedding + Upload ======
def embed_and_upsert(doc_id, chunks, metadata):
    print(f"🧠 Embedding {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks)

    # ✅ Convert to list of Python floats
    vector_avg = [float(x) for x in np.mean(embeddings, axis=0)]

    index.upsert(
        vectors=[{
            "id": doc_id,
            "values": vector_avg,
            "metadata": {
                "filename": f"{doc_id}.pdf",
                **metadata
            }
        }],
        namespace=namespace
    )
    print(f"✅ Upserted vector with ID: {doc_id}")

# ====== Main Process ======
def main():
    doc_id = Path(pdf_path).stem

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return

    print(f"📄 Processing file: {pdf_path}")
    chunks, full_text = read_pdf_chunks(pdf_path)
    metadata = extract_metadata(full_text)
    embed_and_upsert(doc_id, chunks, metadata)
    print("🎉 Insert completed successfully.")

if __name__ == "__main__":
    main()

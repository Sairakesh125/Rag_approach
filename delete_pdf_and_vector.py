import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

# === CONFIG ===
# Set full PDF path here
pdf_path = r"C:\Users\naape\Downloads\Rag_dense_project\data\MHC_CaseStatus_511655.pdf"
# =================

# === LOAD ENV ===
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
namespace = "__default__"

# === Extract doc_id from filename ===
doc_id = Path(pdf_path).stem

# === Step 1: Delete vector from Pinecone ===
def delete_from_pinecone(doc_id):
    index.delete(ids=[doc_id], namespace=namespace)
    print(f"✅ Deleted vector with ID: {doc_id} from Pinecone.")

# === Step 2: Delete PDF file from local ===
def delete_local_pdf(pdf_path):
    pdf_file = Path(pdf_path)
    if pdf_file.exists():
        pdf_file.unlink()
        print(f"✅ Deleted file: {pdf_file}")
    else:
        print(f"⚠️ File not found: {pdf_file}")

# === Run both ===
def main():
    delete_from_pinecone(doc_id)
    delete_local_pdf(pdf_path)
    print("🎉 Done: File + Vector both deleted.")

if __name__ == "__main__":
    main()

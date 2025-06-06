import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY") or "your-api-key-here"
)

# Print available indexes or confirm connection
print("✅ Pinecone initialized successfully!")
print("📂 Indexes:", pc.list_indexes().names())

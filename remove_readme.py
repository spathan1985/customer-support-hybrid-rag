import json
import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("support-rag")

# Load the current local corpus
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Identify README chunks
readme_chunks = [
    chunk for chunk in chunks
    if chunk["source"].endswith("/README.md")
]

print(f"Found {len(readme_chunks)} README chunks")

# Delete their vectors from Pinecone
if readme_chunks:
    index.delete(
        ids=[chunk["id"] for chunk in readme_chunks],
        namespace="support-docs"
    )

# Remove them from the local BM25 corpus
clean_chunks = [
    chunk for chunk in chunks
    if not chunk["source"].endswith("/README.md")
]

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(clean_chunks, f, indent=2, ensure_ascii=False)

print(f"Remaining local chunks: {len(clean_chunks)}")
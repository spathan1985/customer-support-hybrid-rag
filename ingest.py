import os
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPLACE_EXISTING = True
# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

DATA_DIR = Path("data")
INDEX_NAME = "support-rag"
NAMESPACE = "support-docs"

# Nebius embedding configuration
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIMENSIONS = 1024

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Pinecone upload batch size
BATCH_SIZE = 50

# Local output for BM25
CHUNKS_FILE = Path("chunks.json")

# Set to True only when intentionally replacing all
# vectors in the support-docs namespace.
REPLACE_EXISTING = False


# ============================================================
# 2. INITIALIZE CLIENTS
# ============================================================

nebius = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url="https://api.tokenfactory.nebius.com/v1/"
)

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(INDEX_NAME)


# ============================================================
# 3. DOCUMENT LOADING
# ============================================================

def load_document(path):
    """
    Load PDF, Markdown, or text documents.
    """

    if path.suffix.lower() == ".pdf":

        reader = PdfReader(str(path))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    return path.read_text(encoding="utf-8")

def load_faqs(path):
    """
    Load FAQs from a JSONL file.
    Each FAQ becomes one document chunk.
    """
    faq_chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            faq = json.loads(line)

            faq_id = faq["id"]

            text = (
                f"Product: {faq['product']}\n"
                f"Question: {faq['question']}\n"
                f"Answer: {faq['answer']}"
            )

            faq_chunks.append({
                "id": faq_id,
                "text": text,
                "source": str(path),
                "document_type": "faqs",
                "product": faq["product"],
                "chunk_number": 0,
                "line_number": line_number
            })

    return faq_chunks

# ============================================================
# 4. DOCUMENT CHUNKING
# ============================================================

def chunk_text(text):
    """
    Split documents into overlapping chunks while
    respecting natural text boundaries where possible.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


# ============================================================
# 5. GENERATE NEBIUS EMBEDDINGS
# ============================================================

def generate_embedding(text):
    """
    Generate 1024-dimensional embeddings using Nebius.
    """

    response = nebius.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIMENSIONS
    )

    embedding = response.data[0].embedding

    if len(embedding) != EMBED_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBED_DIMENSIONS} dimensions, "
            f"but received {len(embedding)}"
        )

    return embedding


# ============================================================
# 6. PREPARE DOCUMENT CHUNKS
# ============================================================

def prepare_document_chunks():

    all_chunks = []

    supported_extensions = {".pdf", ".txt", ".md"}

    for path in sorted(DATA_DIR.rglob("*")):

        if not path.is_file():
            continue
        
        # NEW: Exclude README files from ingestion
        if path.name.lower() == "readme.md":
            continue

        # Exclude support tickets
        if "ticket" in str(path).lower():
            continue

        # Handle FAQ JSONL files separately
        if path.suffix.lower() == ".jsonl":

            if "faq" in str(path).lower():
                print(f"Processing FAQs: {path}")
                all_chunks.extend(load_faqs(path))

            continue

        # Handle manuals and troubleshooting guides
        if path.suffix.lower() not in supported_extensions:
            continue

        print(f"Processing: {path}")

        text = load_document(path)

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            vector_id = hashlib.sha256(
                f"{path}:{chunk_number}:{chunk}".encode("utf-8")
            ).hexdigest()

            all_chunks.append({
                "id": vector_id,
                "text": chunk,
                "source": str(path),
                "document_type": path.parent.name,
                "chunk_number": chunk_number
            })

    return all_chunks

# ============================================================
# 7. UPLOAD EMBEDDINGS TO PINECONE
# ============================================================

def upload_to_pinecone(chunks):
    """
    Generate embeddings and upload vectors to Pinecone.
    """

    vectors = []

    for i, chunk in enumerate(chunks, start=1):

        embedding = generate_embedding(chunk["text"])

        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
            "text": chunk["text"],
            "source": chunk["source"],
            "document_type": chunk["document_type"],
            "chunk_number": chunk["chunk_number"],
            "source_id": chunk.get("id"),
            "product": chunk.get("product", ""),
            "line_number": chunk.get("line_number", 0)
            }
        })

        if len(vectors) >= BATCH_SIZE:

            index.upsert(
                vectors=vectors,
                namespace=NAMESPACE
            )

            print(f"Uploaded {len(vectors)} vectors")
            vectors = []

        print(f"Embedded {i}/{len(chunks)} chunks", end="\r")

    # Upload remaining vectors
    if vectors:

        index.upsert(
            vectors=vectors,
            namespace=NAMESPACE
        )

        print(f"\nUploaded remaining {len(vectors)} vectors")


# ============================================================
# 8. SAVE CHUNKS FOR BM25
# ============================================================

def save_chunks(chunks):
    """
    Save the exact chunks used for Pinecone ingestion.

    BM25 will use this file to perform keyword search.
    """

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:

        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved {len(chunks)} chunks to {CHUNKS_FILE}"
    )


# ============================================================
# 9. MAIN INGESTION PIPELINE
# ============================================================

def ingest_documents():

    print("\nStarting document ingestion...\n")

    # Step 1: Load and chunk documents
    chunks = prepare_document_chunks()

    print(f"\nTotal chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "No documents found. Check your data directory."
        )

    # Step 2: Verify Pinecone index dimensions
    index_info = pc.describe_index(INDEX_NAME)

    if index_info.dimension != EMBED_DIMENSIONS:
        raise ValueError(
            f"Pinecone index dimension is "
            f"{index_info.dimension}, "
            f"but embedding dimension is "
            f"{EMBED_DIMENSIONS}"
        )

    # Step 3: Optionally replace existing vectors
    if REPLACE_EXISTING:

        print(
            f"\nDeleting existing vectors from "
            f"namespace '{NAMESPACE}'..."
        )

        index.delete(
            delete_all=True,
            namespace=NAMESPACE
        )

    # Step 4: Generate embeddings and upload
    print("\nGenerating embeddings and uploading to Pinecone...\n")

    upload_to_pinecone(chunks)

    # Step 5: Save the same chunks locally for BM25
    save_chunks(chunks)

    # Step 6: Verify Pinecone statistics
    print("\nChecking Pinecone index statistics...")

    stats = index.describe_index_stats()

    print(stats)

    print("\nDocument ingestion completed successfully!")


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    ingest_documents()
import json
import hashlib
from pathlib import Path

from ingest import chunk_text, load_document

DATA_DIR = Path("data")
OUTPUT_FILE = "chunks.json"

def prepare_chunks():
    documents = []

    for path in DATA_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue

        if "ticket" in str(path).lower():
            continue

        text = load_document(path)
        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            vector_id = hashlib.sha256(
                f"{path}:{chunk_number}:{chunk}".encode()
            ).hexdigest()

            documents.append({
                "id": vector_id,
                "text": chunk,
                "source": str(path),
                "document_type": path.parent.name,
                "chunk_number": chunk_number
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    print(f"Saved {len(documents)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_chunks()
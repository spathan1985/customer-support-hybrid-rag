import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Initialize clients
nebius = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url="https://api.tokenfactory.nebius.com/v1/"
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("support-rag")

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIMENSIONS = 1024
NAMESPACE = "support-docs"


def semantic_search(query, top_k=10):

    # Generate query embedding
    response = nebius.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIMENSIONS
    )

    query_embedding = response.data[0].embedding

    # Search Pinecone
    results = index.query(
        namespace=NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Return structured results
    documents = []

    for match in results.matches:

        metadata = match.metadata or {}

        documents.append({
            "id": match.id,
            "score": float(match.score),
            "text": metadata.get("text", ""),
            "source": metadata.get("source", ""),
            "document_type": metadata.get("document_type", "")
        })

    return documents


if __name__ == "__main__":

    query = input("Enter your support question: ")

    results = semantic_search(query, top_k=5)

    for rank, doc in enumerate(results, 1):

        print(f"\nRank: {rank}")
        print(f"Score: {doc['score']:.4f}")
        print(f"Source: {doc['source']}")
        print(f"Text: {doc['text']}")
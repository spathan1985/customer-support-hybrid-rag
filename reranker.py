import os
from dotenv import load_dotenv
from pinecone import Pinecone

from hybrid_search import hybrid_search

load_dotenv()

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

RERANK_MODEL = "bge-reranker-v2-m3"


def rerank_documents(query, documents, top_n=5):
    """
    Rerank hybrid search results using Pinecone.

    Args:
        query: Original customer question
        documents: Results from hybrid search
        top_n: Number of documents to return

    Returns:
        Reranked documents with relevance scores
    """

    if not documents:
        return []

    response = pc.inference.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[
            {"text": doc["text"]}
            for doc in documents
        ],
        rank_fields=["text"],
        top_n=min(top_n, len(documents)),
        return_documents=True
    )

    reranked_results = []

    for item in response.data:

        # Index identifies the original document position
        original_doc = documents[item.index].copy()

        original_doc["rerank_score"] = float(item.score)

        reranked_results.append(original_doc)

    return reranked_results


def hybrid_search_with_reranking(query, top_k=5):

    # Step 1: Retrieve candidates using hybrid search
    hybrid_results = hybrid_search(
        query,
        top_k=40,
        retrieval_k=20
    )

    # Step 2: Rerank and select the top 5 results
    reranked_results = rerank_documents(
        query=query,
        documents=hybrid_results,
        top_n=top_k
    )

    return reranked_results


if __name__ == "__main__":

    query = input("\nEnter your support question: ")

    results = hybrid_search_with_reranking(
        query,
        top_k=5
    )

    print(f"\nTop {len(results)} Reranked Results")

    for rank, doc in enumerate(results, start=1):

        print("\n" + "=" * 65)
        print(f"Rank: {rank}")
        print(f"RRF Score: {doc['rrf_score']:.5f}")
        print(f"Reranker Score: {doc['rerank_score']:.4f}")
        print(f"Retrieved By: {doc['retrieved_by']}")
        print(f"Source: {doc['source']}")
        print(f"Document Type: {doc['document_type']}")
        print(f"\nText:\n{doc['text']}")
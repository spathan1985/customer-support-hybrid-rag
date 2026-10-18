from semantic_search import semantic_search
from bm25_search import keyword_search


# ============================================================
# 1. RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    k=60
):
    """
    Combine semantic and BM25 rankings using RRF.

    RRF Score = sum(1 / (k + rank))

    Rank starts at 1.
    """

    fused_scores = {}
    document_map = {}

    retrieval_methods = {
        "semantic": semantic_results,
        "bm25": bm25_results
    }

    for method, results in retrieval_methods.items():

        for rank, doc in enumerate(results, start=1):

            doc_id = doc["id"]

            # Calculate RRF contribution
            rrf_score = 1 / (k + rank)

            # Accumulate scores across retrieval methods
            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                + rrf_score
            )

            # Preserve document information
            if doc_id not in document_map:
                document_map[doc_id] = {
                    "id": doc_id,
                    "text": doc["text"],
                    "source": doc["source"],
                    "document_type": doc["document_type"],
                    "retrieved_by": []
                }

            document_map[doc_id]["retrieved_by"].append(method)

    # Sort documents by combined RRF score
    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    fused_results = []

    for doc_id in ranked_ids:

        document = document_map[doc_id].copy()

        document["rrf_score"] = fused_scores[doc_id]

        fused_results.append(document)

    return fused_results


# ============================================================
# 2. HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    top_k=20,
    retrieval_k=10
):
    """
    Retrieve documents using semantic search and BM25,
    combine them using RRF, and return top_k results.
    """

    semantic_results = semantic_search(
        query,
        top_k=retrieval_k
    )

    bm25_results = keyword_search(
        query,
        top_k=retrieval_k
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results
    )

    return fused_results[:top_k]


# ============================================================
# 3. TEST HYBRID SEARCH
# ============================================================

if __name__ == "__main__":

    query = input("\nEnter your support question: ")

    results = hybrid_search(query, top_k=10)

    print(f"\nTop {len(results)} Hybrid Search Results")

    for rank, doc in enumerate(results, start=1):

        print("\n" + "=" * 65)
        print(f"Rank: {rank}")
        print(f"RRF Score: {doc['rrf_score']:.5f}")
        print(f"Retrieved By: {doc['retrieved_by']}")
        print(f"Source: {doc['source']}")
        print(f"Document Type: {doc['document_type']}")
        print(f"\nText:\n{doc['text']}")
from semantic_search import semantic_search
from bm25_search import keyword_search

def compare_retrievers(query, top_k=10):

    semantic_results = semantic_search(query, top_k)
    bm25_results = keyword_search(query, top_k)

    semantic_ids = {
        doc["id"] for doc in semantic_results
    }

    bm25_ids = {
        doc["id"] for doc in bm25_results
    }

    overlap = semantic_ids.intersection(bm25_ids)

    print(f"\nQuery: {query}")
    print(f"Semantic results: {len(semantic_ids)}")
    print(f"BM25 results: {len(bm25_ids)}")
    print(f"Common chunks: {len(overlap)}")

    print(
        f"Overlap percentage: "
        f"{len(overlap) / max(len(semantic_ids), 1) * 100:.1f}%"
    )

    print("\nRanking Comparison:")
    print("-" * 75)

    all_ids = semantic_ids.union(bm25_ids)

    for doc_id in all_ids:

        semantic_rank = next(
            (
                i + 1
                for i, doc in enumerate(semantic_results)
                if doc["id"] == doc_id
            ),
            None
        )

        bm25_rank = next(
            (
                i + 1
                for i, doc in enumerate(bm25_results)
                if doc["id"] == doc_id
            ),
            None
        )

        print(
            f"Chunk: {doc_id[:8]} | "
            f"Semantic Rank: {semantic_rank} | "
            f"BM25 Rank: {bm25_rank}"
        )


if __name__ == "__main__":

    query = input("Enter your support question: ")

    compare_retrievers(query)
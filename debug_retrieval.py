from semantic_search import semantic_search
from bm25_search import keyword_search
from hybrid_search import reciprocal_rank_fusion
from reranker import rerank_documents

QUESTION = (
    "How do I set up my Aster Thermostat T100 "
    "for the first time?"
)

GOLD_SOURCE = "aster_thermostat_t100.md"


def inspect_results(method, results):

    print(f"\n{'=' * 65}")
    print(f"{method.upper()} RESULTS")
    print("=" * 65)

    found = False

    for rank, doc in enumerate(results, start=1):

        source = doc["source"]
        is_gold = source.endswith(GOLD_SOURCE)

        if is_gold:
            found = True

        print(f"\nRank: {rank}")
        print(f"Source: {source}")
        print(f"Gold source: {is_gold}")

        if "score" in doc:
            print(f"Score: {doc['score']:.4f}")

        if "rrf_score" in doc:
            print(f"RRF Score: {doc['rrf_score']:.5f}")

        if "rerank_score" in doc:
            print(f"Reranker Score: {doc['rerank_score']:.4f}")

        print(f"Text: {doc['text'][:250]}")

    print(f"\nGold source retrieved: {found}")


def main():

    # Retrieve more candidates for debugging
    semantic = semantic_search(QUESTION, top_k=20)
    bm25 = keyword_search(QUESTION, top_k=20)

    hybrid = reciprocal_rank_fusion(semantic, bm25)

    reranked = rerank_documents(
        query=QUESTION,
        documents=hybrid,
        top_n=len(hybrid)
    )

    inspect_results("Semantic", semantic)
    inspect_results("BM25", bm25)
    inspect_results("Hybrid RRF", hybrid)
    inspect_results("Reranked", reranked)


if __name__ == "__main__":
    main()
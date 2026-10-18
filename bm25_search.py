import json
import re

from rank_bm25 import BM25Okapi

# Load the same chunks used in Pinecone
with open("chunks.json", "r", encoding="utf-8") as f:
    documents = json.load(f)


def tokenize(text):
    """
    Lowercase and tokenize text.
    Preserve alphanumeric error codes such as E204.
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# Tokenize all documents
tokenized_corpus = [
    tokenize(doc["text"])
    for doc in documents
]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)


def keyword_search(query, top_k=5):

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked_indices:

        if scores[i] <= 0:
            continue

        results.append({
            "id": documents[i]["id"],
            "score": float(scores[i]),
            "text": documents[i]["text"],
            "source": documents[i]["source"],
            "document_type": documents[i]["document_type"]
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":

    query = input("Enter your support question: ")

    results = keyword_search(query)

    for rank, result in enumerate(results, 1):

        print(f"\n{'=' * 60}")
        print(f"Rank: {rank}")
        print(f"BM25 Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Document Type: {result['document_type']}")
        print(f"\nText:\n{result['text']}")
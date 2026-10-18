from reranker import hybrid_search_with_reranking
from evidence_validator import validate_evidence


questions = [
    "How do I fix error E204 on my Aster Thermostat T100?",
    "How do I fix error E999 on my EchoLight Bulb B500?",
    "How do I replace the internal circuit board in my Aster Thermostat T100?",
]


for question in questions:

    documents = hybrid_search_with_reranking(
        query=question,
        top_k=5
    )

    result = validate_evidence(
        question=question,
        documents=documents,
        min_rerank_score=0.40
    )

    print("\n" + "=" * 70)
    print("Question:", question)
    print("Sufficient:", result.is_sufficient)
    print("Reason:", result.reason)
    print("Top score:", result.top_score)
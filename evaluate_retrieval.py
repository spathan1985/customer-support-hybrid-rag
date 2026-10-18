# evaluate_retrieval.py

import json
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

from semantic_search import semantic_search
from bm25_search import keyword_search
from hybrid_search import reciprocal_rank_fusion
from reranker import rerank_documents


# ============================================================
# 1. CONFIGURATION
# ============================================================

EVAL_FILE = Path("evaluation.jsonl")
CHUNKS_FILE = Path("chunks.json")
OUTPUT_DIR = Path("evaluation_results")

RETRIEVAL_K = 20
EVAL_K = 5

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD EVALUATION DATA
# ============================================================

def load_evaluation_data():

    records = []

    with open(EVAL_FILE, "r", encoding="utf-8") as f:

        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} evaluation questions")

    return records


# ============================================================
# 3. BUILD SOURCE LOOKUP
# ============================================================

def normalize_path(path):
    """
    Normalize local and gold-source paths.

    Example:
    data/manuals/device.md -> manuals/device.md
    """

    path = Path(path).as_posix().removeprefix("./")

    if path.startswith("data/"):
        path = path[5:]

    return path


def build_source_lookup():
    """
    Map each chunk ID to its logical source.

    FAQs are identified individually.
    Manuals and troubleshooting guides use source paths.
    """

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    lookup = {}

    for chunk in chunks:

        chunk_id = chunk["id"]

        if chunk["document_type"] == "faqs":
            source_key = chunk_id

        else:
            source_key = normalize_path(chunk["source"])

        lookup[chunk_id] = source_key

    return lookup


SOURCE_LOOKUP = build_source_lookup()


# ============================================================
# 4. IDENTIFY GOLD SOURCES
# ============================================================

def get_gold_sources(item):

    gold = set()

    for source in item["gold_sources"]:

        if source["type"] == "faq":
            gold.add(source["source_id"])

        else:
            gold.add(normalize_path(source["path"]))

    return gold


# ============================================================
# 5. IDENTIFY RETRIEVED SOURCES
# ============================================================

def get_retrieved_sources(results, k=5):

    sources = []

    for doc in results[:k]:

        chunk_id = doc["id"]

        if chunk_id not in SOURCE_LOOKUP:
            raise ValueError(
                f"Unknown chunk ID: {chunk_id}. "
                "Check that Pinecone and chunks.json are synchronized."
            )

        sources.append(SOURCE_LOOKUP[chunk_id])

    return sources


# ============================================================
# 6. CALCULATE RETRIEVAL METRICS
# ============================================================

def calculate_metrics(results, gold_sources, k=5):

    retrieved = get_retrieved_sources(results, k)

    if not gold_sources:
        raise ValueError(
            "Retrieval metrics require at least one gold source."
        )

    relevant_found = set(retrieved).intersection(gold_sources)

    # Hit@K
    hit = int(len(relevant_found) > 0)

    # Recall@K
    recall = len(relevant_found) / len(gold_sources)

    # Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, source in enumerate(retrieved, start=1):

        if source in gold_sources:
            reciprocal_rank = 1 / rank
            break

    return {
        "hit_at_5": hit,
        "recall_at_5": recall,
        "rr_at_5": reciprocal_rank
    }


# ============================================================
# 7. EVALUATE RETRIEVAL
# ============================================================

def evaluate():

    dataset = load_evaluation_data()

    retrieval_records = []
    escalation_records = []
    ranking_records = []

    for i, item in enumerate(dataset, start=1):

        question_id = item["question_id"]
        question = item["question"]
        category = item["category"]

        print(
            f"\nEvaluating {i}/{len(dataset)}: "
            f"{question_id}"
        )

        # ----------------------------------------
        # Retrieve once using each method
        # ----------------------------------------

        semantic_results = semantic_search(
            question,
            top_k=RETRIEVAL_K
        )

        bm25_results = keyword_search(
            question,
            top_k=RETRIEVAL_K
        )

        # ----------------------------------------
        # Hybrid RRF
        # ----------------------------------------

        hybrid_results = reciprocal_rank_fusion(
            semantic_results,
            bm25_results
        )

        # ----------------------------------------
        # Rerank all fused candidates
        # ----------------------------------------

        reranked_results = rerank_documents(
            question,
            hybrid_results,
            top_n=len(hybrid_results)
        )

        # ----------------------------------------
        # Save reranker scores for threshold eval
        # ----------------------------------------

        top_score = (
            reranked_results[0]["rerank_score"]
            if reranked_results
            else None
        )

        escalation_records.append({
            "question_id": question_id,
            "category": category,
            "expected_answerability": item[
                "expected_answerability"
            ],
            "should_escalate": item["should_escalate"],
            "top_rerank_score": top_score
        })

        # ----------------------------------------
        # Compare retrieval methods
        # ----------------------------------------

        experiments = {
            "semantic": semantic_results,
            "bm25": bm25_results,
            "hybrid_rrf": hybrid_results,
            "hybrid_reranked": reranked_results
        }

        # Save retrieved rankings for debugging
        for method, results in experiments.items():

            for rank, doc in enumerate(results, start=1):

                ranking_records.append({
                    "question_id": question_id,
                    "method": method,
                    "rank": rank,
                    "chunk_id": doc["id"],
                    "source": SOURCE_LOOKUP.get(
                        doc["id"], ""
                    ),
                    "score": (
                        doc.get("rerank_score")
                        if method == "hybrid_reranked"
                        else doc.get("rrf_score", doc.get("score"))
                    )
                })

        # Retrieval metrics apply only to
        # answerable questions.
        if item["expected_answerability"] != "answerable":
            continue

        gold_sources = get_gold_sources(item)

        for method, results in experiments.items():

            metrics = calculate_metrics(
                results,
                gold_sources,
                k=EVAL_K
            )

            retrieval_records.append({
                "question_id": question_id,
                "category": category,
                "method": method,
                **metrics
            })

    return (
        retrieval_records,
        escalation_records,
        ranking_records
    )


# ============================================================
# 8. SAVE CSV RESULTS
# ============================================================

def save_csv(filename, records):

    if not records:
        return

    path = OUTPUT_DIR / filename

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {path}")


# ============================================================
# 9. AGGREGATE METRICS
# ============================================================

def summarize_results(records):

    grouped = defaultdict(list)

    for record in records:
        grouped[record["method"]].append(record)

    summary = []

    for method, rows in grouped.items():

        summary.append({
            "method": method,
            "questions": len(rows),
            "hit_at_5": mean(
                r["hit_at_5"] for r in rows
            ),
            "recall_at_5": mean(
                r["recall_at_5"] for r in rows
            ),
            "mrr_at_5": mean(
                r["rr_at_5"] for r in rows
            )
        })

    return summary


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    (
        retrieval_records,
        escalation_records,
        ranking_records
    ) = evaluate()

    summary = summarize_results(retrieval_records)

    save_csv(
        "retrieval_details.csv",
        retrieval_records
    )

    save_csv(
        "retrieval_summary.csv",
        summary
    )

    save_csv(
        "escalation_scores.csv",
        escalation_records
    )

    save_csv(
        "retrieved_rankings.csv",
        ranking_records
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    for row in summary:

        print(f"\nMethod: {row['method']}")
        print(f"Questions: {row['questions']}")
        print(f"Hit@5: {row['hit_at_5']:.4f}")
        print(f"Recall@5: {row['recall_at_5']:.4f}")
        print(f"MRR@5: {row['mrr_at_5']:.4f}")

    print("\nEvaluation completed successfully!")
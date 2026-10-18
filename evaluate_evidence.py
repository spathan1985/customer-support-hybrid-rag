import json
import csv
from pathlib import Path
from collections import defaultdict

from evidence_validator import validate_evidence


# ============================================================
# 1. CONFIGURATION
# ============================================================

EVAL_FILE = Path("evaluation.jsonl")
CHUNKS_FILE = Path("chunks.json")

RANKINGS_FILE = Path(
    "evaluation_results/retrieved_rankings.csv"
)

OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_K = 5
THRESHOLD = 0.40


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    return {chunk["id"]: chunk for chunk in chunks}


def load_rankings():
    rankings = defaultdict(list)

    with open(
        RANKINGS_FILE,
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["method"] != "hybrid_reranked":
                continue

            rankings[row["question_id"]].append(row)

    # Preserve reranker ordering
    for question_id in rankings:

        rankings[question_id].sort(
            key=lambda row: int(row["rank"])
        )

    return rankings


# ============================================================
# 3. RECONSTRUCT RERANKED DOCUMENTS
# ============================================================

def reconstruct_documents(
    question_id,
    rankings,
    chunk_lookup
):
    rows = rankings.get(question_id, [])

    documents = []

    for row in rows[:TOP_K]:

        chunk_id = row["chunk_id"]

        if chunk_id not in chunk_lookup:
            raise ValueError(
                f"Chunk {chunk_id} missing from chunks.json"
            )

        chunk = chunk_lookup[chunk_id]

        documents.append({
            "id": chunk_id,
            "text": chunk["text"],
            "source": chunk["source"],
            "document_type": chunk["document_type"],
            "product": chunk.get("product", ""),
            "rerank_score": float(row["score"])
        })

    return documents


# ============================================================
# 4. EVALUATION CONFIGURATIONS
# ============================================================

EXPERIMENTS = {
    "baseline": {
        "enable_threshold": False,
        "enable_error_code_check": False
    },

    "threshold_only": {
        "enable_threshold": True,
        "enable_error_code_check": False
    },

    "combined_validation": {
        "enable_threshold": True,
        "enable_error_code_check": True
    }
}


# ============================================================
# 5. RUN EVIDENCE VALIDATION
# ============================================================

def evaluate():

    dataset = load_jsonl(EVAL_FILE)
    chunk_lookup = load_chunks()
    rankings = load_rankings()

    results = []

    for item in dataset:

        question_id = item["question_id"]
        question = item["question"]

        documents = reconstruct_documents(
            question_id,
            rankings,
            chunk_lookup
        )

        # Fail explicitly if rankings are missing.
        # Do not mistake missing evaluation data
        # for a genuine no-document retrieval result.
        if not documents:
            raise ValueError(
                f"No saved reranked results for {question_id}"
            )

        expected_escalation = item["should_escalate"]

        for experiment, config in EXPERIMENTS.items():

            validation = validate_evidence(
                question=question,
                documents=documents,
                enable_threshold=config["enable_threshold"],
                min_rerank_score=THRESHOLD,
                enable_error_code_check=(
                    config["enable_error_code_check"]
                )
            )

            predicted_escalation = (
                not validation.is_sufficient
            )

            results.append({
                "question_id": question_id,
                "category": item["category"],
                "experiment": experiment,
                "expected_escalation": expected_escalation,
                "predicted_escalation": predicted_escalation,
                "correct": (
                    expected_escalation
                    == predicted_escalation
                ),
                "reason": validation.reason,
                "top_rerank_score": validation.top_score
            })

    return results


# ============================================================
# 6. CALCULATE ESCALATION METRICS
# ============================================================

def calculate_metrics(rows):

    tp = sum(
        r["expected_escalation"]
        and r["predicted_escalation"]
        for r in rows
    )

    fp = sum(
        not r["expected_escalation"]
        and r["predicted_escalation"]
        for r in rows
    )

    fn = sum(
        r["expected_escalation"]
        and not r["predicted_escalation"]
        for r in rows
    )

    tn = sum(
        not r["expected_escalation"]
        and not r["predicted_escalation"]
        for r in rows
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    accuracy = (tp + tn) / len(rows)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


# ============================================================
# 7. SAVE RESULTS
# ============================================================

def save_csv(path, records):

    if not records:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    results = evaluate()

    save_csv(
        OUTPUT_DIR / "evidence_eval_details.csv",
        results
    )

    summary = []

    for experiment in EXPERIMENTS:

        experiment_rows = [
            r for r in results
            if r["experiment"] == experiment
        ]

        metrics = calculate_metrics(experiment_rows)

        summary.append({
            "experiment": experiment,
            **metrics
        })

    save_csv(
        OUTPUT_DIR / "evidence_eval_summary.csv",
        summary
    )

    print("\nEVIDENCE VALIDATION RESULTS")
    print("=" * 70)

    for row in summary:

        print(f"\nExperiment: {row['experiment']}")
        print(f"Precision: {row['precision']:.3f}")
        print(f"Recall: {row['recall']:.3f}")
        print(f"F1: {row['f1']:.3f}")
        print(f"Accuracy: {row['accuracy']:.3f}")

        print(
            f"TP={row['TP']} "
            f"FP={row['FP']} "
            f"FN={row['FN']} "
            f"TN={row['TN']}"
        )

    print("\nEvaluation completed.")
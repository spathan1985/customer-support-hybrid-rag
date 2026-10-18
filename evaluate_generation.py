import json
import time
from pathlib import Path

from generate_answer import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

EVAL_FILE = Path("evaluation.jsonl")
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_FILE = OUTPUT_DIR / "generation_outputs_v2.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

def load_evaluation_data():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


# ============================================================
# LOAD PREVIOUSLY COMPLETED QUESTIONS
# ============================================================

def load_completed_ids():
    if not OUTPUT_FILE.exists():
        return set()

    completed = set()

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                completed.add(record["question_id"])

    return completed


# ============================================================
# GENERATE ANSWERS
# ============================================================

def evaluate_generation():

    dataset = load_evaluation_data()
    completed_ids = load_completed_ids()

    print(f"Total questions: {len(dataset)}")
    print(f"Previously completed: {len(completed_ids)}")

    for i, item in enumerate(dataset, start=1):

        question_id = item["question_id"]

        if question_id in completed_ids:
            print(f"Skipping {question_id}: already completed")
            continue

        question = item["question"]

        print(
            f"\nProcessing {i}/{len(dataset)}: "
            f"{question_id}"
        )

        start_time = time.perf_counter()

        try:
            result = generate_answer(
                question=question,
                enable_threshold=True,
                min_rerank_score=0.40,
                enable_error_code_check=True
            )

            latency = time.perf_counter() - start_time

            record = {
                "question_id": question_id,
                "question": question,
                "category": item["category"],
                "expected_answerability": item[
                    "expected_answerability"
                ],
                "expected_facts": item["expected_facts"],
                "expected_escalation": item["should_escalate"],
                "generated_answer": result["answer"],
                "actual_status": result["status"],
                "validation_reason": result["reason"],
                "top_rerank_score": result["top_rerank_score"],
                "latency_seconds": round(latency, 3),
                "sources": [
                    {
                        "id": doc["id"],
                        "source": doc["source"],
                        "text": doc["text"],
                        "rerank_score": doc.get(
                            "rerank_score"
                        )
                    }
                    for doc in result["sources"]
                ]
            }

            # Append immediately so the run is resumable
            with open(
                OUTPUT_FILE,
                "a",
                encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

            print(f"Status: {result['status']}")
            print(f"Latency: {latency:.2f}s")

        except Exception as e:
            print(f"Error processing {question_id}: {e}")
            print("Question will be retried on the next run.")

    print("\nGeneration run completed.")
    print(f"Results saved to: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_generation()
import json
import time
from pathlib import Path
from statistics import mean

from judge_generation import judge_answer


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("evaluation_results")

VERSIONS = {
    "v1": RESULTS_DIR / "generation_outputs.jsonl",
    "v2": RESULTS_DIR / "generation_outputs_v2.jsonl"
}

JUDGE_OUTPUT = RESULTS_DIR / "generation_comparison_judgments.jsonl"


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


# ============================================================
# RUN SAME JUDGE FOR BOTH VERSIONS
# ============================================================

def run_judgments():

    completed = set()

    if JUDGE_OUTPUT.exists():

        for record in load_jsonl(JUDGE_OUTPUT):
            completed.add(
                (record["version"], record["question_id"])
            )

    for version, filepath in VERSIONS.items():

        records = load_jsonl(filepath)

        print(f"\nEvaluating {version.upper()}")
        print(f"Questions: {len(records)}")

        for record in records:

            question_id = record["question_id"]
            key = (version, question_id)

            if key in completed:
                print(f"Skipping {version}: {question_id}")
                continue

            print(f"Judging {version}: {question_id}")

            try:

                start = time.perf_counter()

                judgment = judge_answer(record)

                latency = time.perf_counter() - start

                output = {
                    "version": version,
                    "question_id": question_id,
                    "category": record["category"],
                    "expected_answerability":
                        record["expected_answerability"],
                    "actual_status": record["actual_status"],
                    "judgment": judgment,
                    "judge_latency_seconds": round(latency, 3)
                }

                with open(
                    JUDGE_OUTPUT,
                    "a",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        json.dumps(output, ensure_ascii=False)
                        + "\n"
                    )

            except Exception as e:
                print(f"Error: {e}")


# ============================================================
# SUMMARIZE JUDGMENTS
# ============================================================

def summarize():

    records = load_jsonl(JUDGE_OUTPUT)

    for version in VERSIONS:

        # Evaluate generation quality on answerable
        # questions only.
        rows = [
            r for r in records
            if r["version"] == version
            and r["expected_answerability"] == "answerable"
        ]

        if not rows:
            continue

        fact_scores = []
        citation_scores = []
        correctness = []
        unsupported_counts = []

        for row in rows:

            judgment = row["judgment"]

            facts = judgment.get("fact_results", [])

            if facts:
                fact_scores.append(
                    sum(
                        bool(f.get("covered", False))
                        for f in facts
                    ) / len(facts)
                )

            citations = judgment.get(
                "citation_results", []
            )

            if citations:
                citation_scores.append(
                    sum(
                        bool(c.get("supported", False))
                        for c in citations
                    ) / len(citations)
                )

            correctness.append(
                int(judgment.get("overall_correct", False))
            )

            unsupported_counts.append(
                len(judgment.get("unsupported_claims", []))
            )

        print("\n" + "=" * 60)
        print(f"GENERATION {version.upper()} RESULTS")
        print("=" * 60)

        print("Questions evaluated:", len(rows))

        print(
            "Average fact coverage:",
            round(mean(fact_scores), 3)
            if fact_scores else "N/A"
        )

        print(
            "Average citation support:",
            round(mean(citation_scores), 3)
            if citation_scores else "N/A"
        )

        print(
            "Overall correctness:",
            round(mean(correctness), 3)
        )

        print(
            "Average unsupported claims per answer:",
            round(mean(unsupported_counts), 3)
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_judgments()
    summarize()
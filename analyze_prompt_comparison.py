import json
from pathlib import Path

FILE = Path(
    "evaluation_results/generation_comparison_judgments.jsonl"
)

with open(FILE, encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

results = {}

for record in records:
    question_id = record["question_id"]
    version = record["version"]

    results.setdefault(question_id, {})[version] = {
        "correct": record["judgment"]["overall_correct"],
        "unsupported": len(
            record["judgment"].get("unsupported_claims", [])
        )
    }

print("\nQUESTION-LEVEL COMPARISON")
print("=" * 70)

for question_id, versions in sorted(results.items()):

    if "v1" not in versions or "v2" not in versions:
        continue

    v1 = versions["v1"]
    v2 = versions["v2"]

    if not v1["correct"] and v2["correct"]:
        status = "IMPROVED"

    elif v1["correct"] and not v2["correct"]:
        status = "REGRESSED"

    else:
        status = "UNCHANGED"

    print(
        f"{question_id} | "
        f"V1: {v1['correct']} | "
        f"V2: {v2['correct']} | "
        f"{status}"
    )
import json
from pathlib import Path
from collections import Counter

path = Path("evaluation_results/generation_outputs.jsonl")

with open(path, encoding="utf-8") as f:
    results = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print("Total generated records:", len(results))

print("Status distribution:")
print(Counter(r["actual_status"] for r in results))

print("\nSample answer:")
print(results[0]["generated_answer"])
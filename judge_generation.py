import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "evaluation_results/generation_outputs.jsonl"
)

OUTPUT_FILE = Path(
    "evaluation_results/generation_judgments.jsonl"
)

JUDGE_MODEL = os.environ["NEBIUS_CHAT_MODEL"]

client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url="https://api.tokenfactory.nebius.com/v1/"
)


# ============================================================
# LOAD RECORDS
# ============================================================

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_completed_ids():
    if not OUTPUT_FILE.exists():
        return set()

    return {
        record["question_id"]
        for record in load_jsonl(OUTPUT_FILE)
    }


# ============================================================
# FORMAT SUPPORTING SOURCES
# ============================================================

def format_sources(sources):
    parts = []

    for i, source in enumerate(sources, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"Document: {source['source']}\n"
            f"Content:\n{source['text']}"
        )

    return "\n\n".join(parts)


# ============================================================
# JUDGE ANSWER
# ============================================================

def judge_answer(record):

    system_prompt = """
You are an evaluator of a customer support RAG application.

Evaluate the generated answer using ONLY:
1. The customer question.
2. The expected facts.
3. The retrieved documentation.
4. The generated answer.

Do not use external knowledge.

Evaluate these dimensions:

FACT COVERAGE:
For every expected fact, determine whether the generated
answer correctly communicates that fact.

FAITHFULNESS:
Identify factual claims in the generated answer that are
not supported by the retrieved documentation.

CITATION ACCURACY:
Check whether citations such as [Source 1] refer to
provided sources and support the associated claims.

Do not assume a citation is correct merely because
the source exists.

Do not penalize an answer for omitting irrelevant details.

Return ONLY valid JSON with this structure:

{
  "fact_results": [
    {
      "fact": "Expected fact",
      "covered": true,
      "explanation": "Brief explanation"
    }
  ],
  "unsupported_claims": [
    {
      "claim": "Unsupported statement",
      "reason": "Why it is unsupported"
    }
  ],
  "citation_results": [
    {
      "citation": "[Source 1]",
      "supported": true,
      "explanation": "Brief explanation"
    }
  ],
  "overall_correct": true,
  "notes": "Brief summary"
}

Rules:
- Return one fact_results entry per expected fact.
- Use boolean values, not strings.
- Mark unsupported or incorrect claims explicitly.
- Set overall_correct to false if an important expected
  fact is missing, a factual claim is incorrect, or
  the answer contains a material unsupported claim.
"""

    user_prompt = f"""
CUSTOMER QUESTION:
{record['question']}

EXPECTED FACTS:
{json.dumps(record['expected_facts'], indent=2)}

RETRIEVED DOCUMENTATION:
{format_sources(record['sources'])}

GENERATED ANSWER:
{record['generated_answer']}

EXPECTED ANSWERABILITY:
{record['expected_answerability']}

ACTUAL STATUS:
{record['actual_status']}
"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        max_tokens=2000
    )

    content = response.choices[0].message.content

    # Remove Markdown code fences if the model adds them
    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0].strip()

    return json.loads(content)


# ============================================================
# RUN EVALUATION
# ============================================================

def evaluate():

    records = load_jsonl(INPUT_FILE)
    completed_ids = load_completed_ids()

    print(f"Total records: {len(records)}")
    print(f"Previously evaluated: {len(completed_ids)}")

    for i, record in enumerate(records, start=1):

        question_id = record["question_id"]

        if question_id in completed_ids:
            print(f"Skipping {question_id}")
            continue

        print(
            f"\nJudging {i}/{len(records)}: {question_id}"
        )

        try:
            start = time.perf_counter()

            judgment = judge_answer(record)

            latency = time.perf_counter() - start

            output = {
                "question_id": question_id,
                "category": record["category"],
                "expected_answerability": record[
                    "expected_answerability"
                ],
                "actual_status": record["actual_status"],
                "judgment": judgment,
                "judge_latency_seconds": round(latency, 3)
            }

            with open(
                OUTPUT_FILE,
                "a",
                encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(output, ensure_ascii=False)
                    + "\n"
                )

            print(
                f"Overall correct: "
                f"{judgment['overall_correct']}"
            )

        except Exception as e:
            print(f"Error evaluating {question_id}: {e}")
            print("Will retry on the next run.")

    print("\nJudging completed.")


if __name__ == "__main__":
    evaluate()
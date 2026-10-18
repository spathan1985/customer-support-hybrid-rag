import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_sufficient: bool
    reason: str
    top_score: float | None


# Match error codes such as E204, E310, E999
ERROR_CODE_PATTERN = re.compile(
    r"\bE\d{3,4}\b",
    re.IGNORECASE
)


def extract_error_codes(text):
    """Extract normalized error codes."""
    return set(
        code.upper()
        for code in ERROR_CODE_PATTERN.findall(text)
    )


def validate_evidence(
    question,
    documents,
    enable_threshold=True,
    min_rerank_score=0.40,
    enable_error_code_check=True
):
    """
    Validate retrieved evidence before LLM generation.

    Checks:
    1. Documents exist.
    2. Top reranker score meets threshold.
    3. Explicit error codes are supported.
    """

    if not documents:
        return ValidationResult(
            is_sufficient=False,
            reason="no_documents",
            top_score=None
        )

    top_score = max(
        float(doc.get("rerank_score", 0))
        for doc in documents
    )

    # Check 1: Relevance threshold
    if enable_threshold and top_score < min_rerank_score:
        return ValidationResult(
            is_sufficient=False,
            reason="low_relevance",
            top_score=top_score
        )

    # Check 2: Exact error-code validation
    if enable_error_code_check:

        query_codes = extract_error_codes(question)

        if query_codes:

            # Look for codes in retrieved evidence
            document_codes = set()

            for doc in documents:
                document_codes.update(
                    extract_error_codes(doc.get("text", ""))
                )

            missing_codes = query_codes - document_codes

            if missing_codes:
                return ValidationResult(
                    is_sufficient=False,
                    reason=(
                        "unsupported_error_code: "
                        + ", ".join(sorted(missing_codes))
                    ),
                    top_score=top_score
                )

    return ValidationResult(
        is_sufficient=True,
        reason="passed",
        top_score=top_score
    )
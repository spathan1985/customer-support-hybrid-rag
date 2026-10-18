# Hybrid RAG customer support evaluation

This pack contains 30 source-verified questions for the existing synthetic smart-home support corpus: 10 direct, 10 paraphrased, 5 exact error-code, and 5 unanswerable questions. Each of the five products has two direct questions, two paraphrases, one error-code question, and one unanswerable question. Use this small set to diagnose retrieval, grounding, and escalation behavior; it is not a production benchmark.

## Files and provenance

- `evaluation.jsonl`: one question per line, with gold source identities, expected facts, evidence excerpts, and escalation expectations.
- `reference_corpus/`: original manuals, troubleshooting guides, FAQ JSONL, and document manifest copied byte-for-byte from the archive. Tickets are excluded.
- `corpus_manifest.json`: archive SHA-256 and individual file checksums for reproducibility.
- `validate_dataset.py`: dependency-free Python 3.9+ validator.
- `validation_report.json`: validation result for this release.

The source archive `synthetic_support_rag_dataset.zip` was found in the user's Downloads folder and inspected on October 18, 2026. Labels come from its actual contents, not inferred filenames or the earlier chat's examples. The archive has 5 manuals, 40 troubleshooting guides, and 60 individual FAQ records: 105 logical documents in 46 indexable files. The accompanying document manifest lists only the 45 Markdown documents; FAQ IDs come from the JSONL records themselves.

The project's reported approximately 51 chunks is not a verified count of this archive's logical documents. Chunk count depends on ingestion. Confirm that all 60 FAQ records are represented and that each chunk retains its source identity. No Pinecone or BM25 index was accessed or changed while creating this pack.

## Run validation

Unzip the pack, open its folder, and run:

```bash
python3 validate_dataset.py
```

Exit 0 means structural checks passed; exit 1 means a failure. The script checks category and product coverage, question uniqueness, paired labels, all five error codes, source paths and IDs, FAQ line numbers, exact evidence excerpts, corpus hashes, and exclusion of unexpected files from the reference corpus. It reports the number of distinct gold sources covered. These checks do not establish semantic correctness by themselves; expected facts were also reviewed against the documents during labeling.

To check a local copy of the same source version, point to the directory containing `manuals/`, `troubleshooting/`, `faqs/`, and `manifest.json`:

```bash
python3 validate_dataset.py --corpus-root /path/to/clean/corpus
```

A directory containing tickets or other extra files fails the strict inventory check. Use a separate clean corpus directory; do not delete the original tickets. Any changed source file fails the checksum check and requires label review before updating the provenance manifest.

Optionally export every indexed chunk's normalized source path as a JSON array, for example `["manuals/aster_thermostat_t100.md", "faqs/faqs.jsonl"]`, and run:

```bash
python3 validate_dataset.py --indexed-sources indexed_sources.json
```

The example array is illustrative and incomplete. A real export must contain all indexed chunks. Duplicates are allowed. Unknown paths, tickets, evaluation files, and missing source files cause failure. This audits the supplied path inventory only; it cannot prove that chunk content excludes copied tickets or that all FAQ records survived ingestion.

## Label meanings

`question_id` is a stable evaluation ID. `category` has four values: `direct_answerable`, `paraphrased`, `exact_error_code`, and `unanswerable`. `product` identifies the requested device. `expected_answerability` concerns whether the corpus supports the requested substantive answer.

`gold_sources` contains verified alternatives for the required core facts, not a demand to retrieve every listed source. Each entry has a corpus-relative `path`, original `source_id`, document `type`, `product`, and verbatim `evidence`. FAQ entries also have a one-based `line`. The required core facts are in `expected_facts`; accept faithful paraphrases. Gold alternatives are curated, not exhaustive relevance judgments. A different passage may merit credit after manual review if it supports all required facts for the correct product.

`escalation_expectation` is `not_required`, `conditional`, or `required`. `should_escalate` means immediate escalation, and is true only for `required`. An answerable troubleshooting question should receive the supported first steps, with support recommended if they fail or the documentation identifies visible damage as an escalation trigger. An informational setup, access-sharing, or warranty question does not require a referral. Conditional escalation guidance may require a troubleshooting source beyond a minimal manual excerpt; evaluate its citation against the actual returned context.

`intent_group` links each direct question with its paraphrase. `label_status` is `verified_against_archive`; this does not claim verification against your deployed index. `label_notes` records boundary decisions. `context_sources` can support a refusal or safety statement without answering the requested procedure. These are not positive retrieval targets for the missing answer.

## Unanswerable judgments

All 105 indexable logical documents were inspected for coverage. The five unsupported requests concern internal circuit-board replacement, cloud storage retention and pricing, purifier room-area ratings, maximum plug wattage, and an undocumented E999 bulb code. The corpus does not supply these answers. These judgments apply only to the archived version.

Unanswerable rows deliberately have empty `gold_sources`; an absence cannot be represented by a fabricated supporting document. The circuit-board row includes the thermostat manual under `context_sources` because it explicitly says not to disassemble the device. Do not infer that internal parts are inherently nonreplaceable. For E999, do not substitute E512's meaning or resolution. A transparent statement that the requested information is missing, followed by a support referral, is the expected behavior. That referral is the evaluation policy for insufficient evidence, not a claim that every missing specification has its own escalation instruction in the corpus.

## Prevent leakage

Use the same allowlist for Pinecone and BM25: `manuals/*.md`, `troubleshooting/*.md`, and the records in `faqs/faqs.jsonl`. Exclude original `tickets.csv`, this README, manifests, evaluation questions, expected facts, evidence annotations, and result logs. The original tickets contain expected resolutions and source IDs. This pack neither copies nor uses their contents to generate labels.

Keep the evaluation folder outside recursively indexed directories. If the reference corpus is used for indexing, select only its three indexable locations. Index each FAQ as an identifiable record, or preserve which FAQ IDs occur in each chunk. If the current indexes already contain tickets or evaluation answers, rebuild clean indexes before measuring. No live index cleanup is claimed here.

Normalize retrieval paths explicitly: a live path such as `data/troubleshooting/TSG-aster_thermostat_t100-01.md` maps to `troubleshooting/TSG-aster_thermostat_t100-01.md`. Strip only a known ingestion root. Do not match by basename alone. Preserve the original FAQ ID; matching merely `faqs/faqs.jsonl` would incorrectly credit any FAQ chunk for every FAQ question.

## Evaluation procedure

1. Freeze source version, chunking, embedding model, reranker model, candidate budgets, RRF settings, generation prompt, and model parameters. Record actual model IDs from your existing configuration. Keep the corpus identical across methods.
2. Run dense retrieval, BM25, RRF hybrid retrieval, and RRF plus reranking for each question. Cache ranked chunk IDs, normalized source paths, FAQ IDs, chunk text, rank, retrieval method, RRF score, reranker score, latency, and configuration. RRF scores and reranker scores are different quantities.
3. Retrieve at least five results when measuring at five. The generation pipeline's `TOP_K=3` must not silently truncate retrieval evaluation. Compare generation context sizes of three and five separately if useful.
4. Evaluate retrieval on the 25 answerable rows. At a fixed cutoff, Hit@k is 1 if any verified relevant source appears, otherwise 0. MRR@k is the reciprocal rank of the first relevant result, or 0 if none appears. Macro-average by question and report category and product breakdowns. Report direct/paraphrase paired differences as a robustness check.
5. For document-level Recall@k, deduplicate ranked results by logical source ID and compute the fraction of listed gold alternatives retrieved. Report this separately from Hit@k: retrieving one sufficient answer source is not the same as retrieving every equivalent source. The curated gold list makes this recall a diagnostic, not an exhaustive relevance measure. For chunk-level scoring, require the returned text to contain sufficient evidence; matching a manual ID alone can credit the wrong chunk. Expand gold judgments through documented manual adjudication before final comparisons.
6. Evaluate generated answers for each expected fact, unsupported claims, correct product, useful ordering of steps, and citations that actually support their associated claims. Score fact coverage as supported required facts divided by required facts, and separately report the proportion of answers with unsupported claims. For unanswerable rows, assess appropriate abstention, no invented answer, and support referral; skip positive-source retrieval metrics.
7. Establish a threshold-disabled baseline. Cache the same reranked candidates once, then simulate threshold decisions offline. The decision is immediate escalation when no documents exist or the top reranker score is below the candidate threshold. Missing or nonfinite scores are run errors to inspect, not automatic low scores. Report actual no-document cases separately.
8. Compare candidate thresholds, including the earlier 0.1 and 0.5 experiments, only if those values are meaningful for the current reranker's scale. Scores are not confidence probabilities. High relevance does not establish evidence sufficiency: the negative specification questions are intended to test this. Reuse the baseline answer for rows a simulated threshold would allow, and substitute the escalation response for rejected rows. Record LLM-initiated abstention separately from threshold rejection.

For escalation metrics, the positive class is the five rows with `should_escalate=true`. Precision = correctly escalated unanswerable rows / all immediately escalated rows. Recall = correctly escalated unanswerable rows / all unanswerable rows. False escalation rate = immediately escalated answerable rows / 25. False answer rate = substantively answered unanswerable rows / 5. Answer coverage = substantive answers / 30. A referral appended to an invented answer still counts as a false answer. Report counts with each rate, and use undefined rather than zero for a zero denominator. Conditional support advice after valid troubleshooting is not immediate escalation.

## Development and held-out use

This pack is an exploratory development set; no rows are designated as an untouched test set. Ten direct/paraphrase pairs intentionally share facts and must stay together when splitting. Template repetition also creates overlap across products. Random row splitting would overstate generalization.

For a final threshold decision, create additional independently authored questions with separate intent groups, seal a held-out set before tuning, and review its labels against the same source version. If forced to split these 30 rows, keep intent groups intact and preferably hold out whole products, but disclose shared templates and the very small number of negative examples. Do not tune on all 30 and report the same rows as independent test performance. No retrieval or generation scores are included in this delivery because the live pipeline was not run.

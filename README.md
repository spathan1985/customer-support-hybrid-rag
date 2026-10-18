# Customer Support RAG

A Streamlit support assistant for fictional smart-home products. It combines semantic search and keyword search, reranks the results, checks the available evidence, and generates answers with source references. The included documents and evaluation questions are synthetic; this is a learning and evaluation project.

## How it works

1. `ingest.py` reads manuals, troubleshooting guides, and individual FAQ records from `data/`. Text documents use 1,000-character chunks with 150-character overlap.
2. Nebius generates 1,024-dimensional `Qwen/Qwen3-Embedding-8B` embeddings. Pinecone stores the vectors in the `support-rag` index and `support-docs` namespace. The same chunks are saved in `chunks.json` for BM25 keyword search.
3. Semantic search and BM25 results are merged using reciprocal rank fusion (RRF).
4. Pinecone's `bge-reranker-v2-m3` reranks the candidates.
5. Evidence validation checks relevance and explicit error codes. Insufficient evidence produces an escalation response; otherwise, a Nebius chat model generates the answer.

The UI displays retrieved sources and latency, and lets you adjust the relevance threshold (default `0.40`) and error-code validation.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python and dependency management.
- Python 3.11 or newer. `.python-version` selects Python 3.14 to match the existing development environment; uv can install it if needed.
- A Nebius API key with access to the embedding and chat models.
- A Pinecone API key with access to serverless indexes and the reranking model.

This is a **uv application project**: dependencies are declared in `pyproject.toml`, with exact resolutions in `uv.lock`. It keeps the existing script layout and is not configured to build a distributable wheel. Run all commands below from the project root because scripts use relative data paths.

## Setup

```bash
uv sync --locked
cp .env.example .env
```

If you already have a `.env`, keep it instead of copying over it. Edit `.env` with your own credentials:

| Variable | Purpose |
| --- | --- |
| `NEBIUS_API_KEY` | Authentication for embeddings, generation, and judging |
| `NEBIUS_CHAT_MODEL` | Exact chat model ID available in your Nebius account |
| `PINECONE_API_KEY` | Authentication for vector storage, search, and reranking |
| `PINECONE_INDEX` | Index name used by `create_index.py`; keep `support-rag` |

The example chat model ID is a starting configuration; confirm it is available in your account. The embedding endpoint is configured in code as `https://api.tokenfactory.nebius.com/v1/`. `.env` and the virtual environment are excluded by `.gitignore`.

## Prepare the index

These commands call external services and may incur usage charges:

```bash
uv run create_index.py
uv run ingest.py
```

Index creation uses AWS `us-east-1`, cosine similarity, and 1,024 dimensions. Wait for a newly created index to be ready before ingestion. Ingestion and semantic search currently hard-code `support-rag` and `support-docs`; changing only `PINECONE_INDEX` does not reconfigure the full pipeline.

Ingestion excludes README files and paths containing `ticket`, and handles FAQ JSONL records separately. Keep evaluation data outside `data/` to avoid indexing answers. See [the dataset notes](data/README.md).

Use `ingest.py` to regenerate `chunks.json` and upload the matching vectors. The older `prepare_bm25.py` uses different document-selection rules and omits FAQs, so it should not be used to rebuild chunks for the main pipeline. The bundled `chunks.json` does not provision a remote Pinecone index.

The effective `REPLACE_EXISTING` setting in `ingest.py` is `False`: ingestion upserts without clearing the namespace. If documents change, old vectors may remain. Setting it to `True` explicitly deletes all vectors in `support-docs` before uploading the new corpus.

## Run the assistant

```bash
uv run streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Ask a support question or enter an error code. The sidebar controls evidence checks, and each answer can show its retrieved sources.

For interactive command-line exploration:

```bash
uv run bm25_search.py
uv run semantic_search.py
uv run hybrid_search.py
uv run reranker.py
uv run generate_answer.py
```

BM25 runs locally using `chunks.json`; the other commands require the configured services.

## Evaluation

The project includes a 30-question evaluation pack covering direct questions, paraphrases, exact error codes, and unanswerable requests. See [the evaluation guide](hybrid_rag_evaluation/README.md) for provenance, label definitions, and limitations.

Validate the bundled reference dataset locally:

```bash
uv run hybrid_rag_evaluation/validate_dataset.py
```

The validator checks the exact reference-corpus inventory, including unexpected hidden files. If macOS has added a `.DS_Store`, validation fails until that metadata file is moved out of `hybrid_rag_evaluation/reference_corpus/`. A clean copy can also be checked with `--corpus-root /path/to/clean/reference_corpus`.

With the live index prepared, run retrieval, evidence, and generation evaluation in this order:

```bash
uv run evaluate_retrieval.py
uv run evaluate_evidence.py
uv run evaluate_generation.py
```

The scripts read the root `evaluation.jsonl` and write under `evaluation_results/`. Evidence evaluation consumes the retrieval rankings. Generation evaluation writes `generation_outputs_v2.jsonl` and resumes by skipping already recorded question IDs; archive prior results before starting a fresh experiment.

`judge_generation.py` reads the baseline `generation_outputs.jsonl`, while `compare_generation.py` requires both baseline and v2 outputs. Check their input/output constants before judging a new run. Judging uses the configured Nebius chat model and incurs additional API calls. Existing results are saved experiments, not verification of your current environment.

`test.py` and `test_evidence_validator.py` are live-service smoke scripts, not an isolated unit-test suite.

## Project map

| File or directory | Role |
| --- | --- |
| `app.py` | Streamlit chat interface |
| `create_index.py`, `ingest.py` | Index setup and shared corpus ingestion |
| `semantic_search.py`, `bm25_search.py` | Dense and keyword retrieval |
| `hybrid_search.py`, `reranker.py` | Rank fusion and reranking |
| `evidence_validator.py`, `generate_answer.py` | Evidence checks and answer generation |
| `data/` | Synthetic support corpus and dataset notes |
| `chunks.json` | Local copy of indexed chunks used by BM25 |
| `evaluation.jsonl`, `hybrid_rag_evaluation/` | Evaluation questions, reference corpus, and validator |
| `evaluate_*.py`, `judge_generation.py`, `compare_generation.py` | Evaluation and comparison scripts |
| `evaluation_results/` | Saved rankings, answers, and evaluation reports |

## Dependency maintenance

```bash
uv add <package-name>       # Update pyproject.toml, uv.lock, and the environment
uv lock --upgrade          # Resolve dependency upgrades
uv sync --locked           # Install the committed resolution
uv lock --check            # Verify the lockfile matches the project
```

Keep `pyproject.toml`, `uv.lock`, and `.python-version` together in version control. No virtual-environment activation or separate requirements file is needed when using `uv run`.

## Troubleshooting

- **Missing environment variable:** fill in all required values in `.env` and run from the project root.
- **Missing `chunks.json` or empty search results:** run ingestion and confirm the same corpus is present locally and in the `support-docs` namespace.
- **Pinecone dimension mismatch:** the index must use 1,024 dimensions to match the embedding configuration.
- **Model or authentication errors:** confirm your keys and model access, including the exact `NEBIUS_CHAT_MODEL` value.
- **Unexpected escalation:** inspect retrieved sources, reranker scores, and error-code coverage using the UI controls or `debug_retrieval.py`.

# Synthetic customer-support RAG dataset

All brands, devices, procedures, and policies are fictional. Do not use these instructions for real hardware.

Contents: 5 Markdown product manuals, 40 Markdown troubleshooting guides, 60 FAQ JSONL records, 150 CSV tickets, and a document manifest.

**Index:** manuals/, troubleshooting/, and faqs/faqs.jsonl. **Do not index tickets.csv** for the baseline evaluation: tickets include expected answers and would cause leakage.

Tickets include expected_source_id and expected_resolution for evaluation, plus dev/test split. Some tickets request escalation.

Caveat: This is template-generated data with deliberately repetitive wording. Use it to validate the pipeline, not to claim production-level retrieval performance. Improve realism with independently written paraphrases and harder negatives.

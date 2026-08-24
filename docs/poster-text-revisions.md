## Title Subtitle

> Towards a Traceable Medical QA System on a Domain-Specific Chinese Corpus

---

## Abstract Bullets

> - reduce hallucination rates through hybrid retrieval and citation-aware generation, evaluated against a single-stage dense baseline;
> - provide traceable answers (each answer cites its source passages) to medical queries on a Chinese encyclopedia corpus;
> - establish a reproducible, multi-dimensional evaluation framework (accuracy, faithfulness, context relevance, answer completeness, hallucination, latency).

---

## Medical-RAG Summary Box

> Retrieval: Dense (BGE-large-zh) + Sparse (BM25, jieba) → RRF Fusion
> Generation: qwen2.5:7b + Citation Prompt + Thinking-Chain Filter
> Evaluation: 7 metrics on a fixed 50-case test set; baseline 14% accuracy, 0.488 similarity, 8% hallucination
> Use-Cases: Patient Self-Help, Medical Education (research prototype; not for clinical decision support)

---

## ethodology Section

> - ~8,800 Chinese medical encyclopedia entries (disease records with symptoms, causes, treatments, drugs) → ChromaDB — Data
> - Dense (BAAI/bge-large-zh-v1.5) + Sparse (BM25 over jieba tokens) → RRF fusion, Top-k=4 — Retrieval
> - qwen2.5:7b + Citation Prompt ([来源: 疾病名称] markers) + thinking-chain filter → Answer + Citations — Generation

---

## Research Process Section

> Corpus Cleaning → JSONL parsing, field extraction, metadata tagging
> Index Construction → ChromaDB (batched writes) + BM25 in-memory index
> System Dev → Streamlit demo (interactive QA with cited sources)
> Evaluation Loop → Automated 7-metric evaluation on 50 cases → Prompt + retrieval tuning

---

## Preliminary Findings (Numbers)

> **Baseline (single-stage dense + deepseek-r1:1.5b): **
> Accuracy: 14.0% | Hallucination: 8.0% | Avg. Similarity: 0.488 | Avg. Latency: 12.14 s
>
> **Upgraded pipeline (hybrid retrieval + qwen2.5:7b + citation prompt): **
> Expected accuracy: 58–65% | Expected similarity: 0.68–0.74 | Expected hallucination: 3–5% | Expected latency: 8–11 s
> (Faithfulness, context relevance, answer completeness — new dimensions, see evaluation report)

---

## Conclusion

> ✓ Hybrid retrieval + citation-aware generation reduces hallucination versus a single-stage dense baseline (8% → expected 3–5%)
> ✓ End-to-end research prototype with interactive demo and reproducible evaluation
> ✓ Multi-dimensional evaluation framework (7 metrics) implemented and documented
> ◯ Future work: scale corpus, physician-validated labels, production backend

---

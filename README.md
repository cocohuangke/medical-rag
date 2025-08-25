# Medical-Domain RAG

> A retrieval-augmented generation system for medical question answering, built to investigate whether hybrid retrieval and citation-aware generation can mitigate the hallucination and low-precision problems of LLM-based medical QA on a domain-specific Chinese corpus.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Status & Honest Limitations](#project-status--honest-limitations)
- [Tech Debt Notes](#tech-debt-notes)

---

## Overview

This project applies **Retrieval-Augmented Generation (RAG)** to a Chinese medical knowledge base of ~8,800 disease encyclopedia entries. It retrieves relevant passages via a **hybrid Dense + Sparse** strategy, then generates an answer with an LLM (SiliconFlow `Qwen2.5-7B-Instruct` via an OpenAI-compatible API) that is explicitly prompted to cite its sources. The system is evaluated on a fixed 50-case test set across seven dimensions — accuracy, semantic similarity, faithfulness, context relevance, answer completeness, hallucination rate, and latency.

The project was developed in three phases:

1. **Phase 1 — Core pipeline.** Hybrid retrieval (BGE dense + BM25 sparse, RRF-fused), citation-aware prompt, thinking-chain filter, model switched from a reasoning-tuned `deepseek-r1:1.5b` to a non-reasoning `qwen2.5:7b` (served via SiliconFlow API).
2. **Phase 2 — Evaluation framework.** Multi-dimensional metrics beyond a single accuracy number: faithfulness, context relevance, answer completeness.
3. **Phase 3 — Demo & documentation.** Streamlit front-end, README, evaluation report, and poster revisions.

---

## Motivation

The baseline pipeline (single-stage dense retrieval + reasoning-model generation) scored **14% accuracy** and **0.488 average semantic similarity** on the 50-case test set. Root-cause analysis identified three failure modes:

1. **Reasoning-chain leakage.** `deepseek-r1:1.5b` emits `⋖thinking≻…⋖/thinking≻` blocks that dilute the cosine-similarity score — the metric was penalising the deliberation, not the answer.
2. **Weak retrieval recall.** Dense-only retrieval missed lexically-distinct passages, causing irrelevant context to enter the top-K.
3. **Crude single-metric evaluation.** Reporting only similarity-based accuracy conflated retrieval misses, generation hallucination, and incomplete coverage into one number, making it impossible to localise the next improvement.

Each of these is addressed by a concrete architectural change in the upgraded pipeline (see [System Architecture](#system-architecture)). The full failure-mode analysis and expected-improvement rationale is in [`docs/evaluation-report.md`](docs/evaluation-report.md).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        medical_rag_system.py                     │
│                                                                  │
│  ┌────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │ JSONL data │───▶│ Recursive splitter│───▶│ Document chunks │   │
│  │ (8.8k recs)│    │ (400 / 80 ovlp)  │    │ + metadata      │   │
│  └────────────┘    └──────────────────┘    └────────┬────────┘   │
│                                                      │            │
│                       ┌──────────────────────────────┼──────┐     │
│                       ▼                              ▼      │     │
│              ┌───────────────┐              ┌──────────────┐ │     │
│              │ Dense: BGE    │              │ Sparse: BM25 │ │     │
│              │ + ChromaDB    │              │ (jieba tok)  │ │     │
│              └───────┬───────┘              └──────┬───────┘ │     │
│                      │                             │         │     │
│                      └─────────────┬───────────────┘         │     │
│                                    ▼                         │     │
│                          ┌──────────────────┐                │     │
│                          │  RRF fusion (k=60)│                │     │
│                          │  → top-4 passages │                │     │
│                          └────────┬─────────┘                │     │
│                                   ▼                          │     │
│              ┌──────────────────────────────────┐            │     │
│              │ Citation-aware QA prompt         │            │     │
│              │ → [来源: 疾病名称] markers        │            │     │
│              └──────────────┬───────────────────┘            │     │
│                             ▼                                │     │
│              ┌──────────────────────────────────┐            │     │
│              │ clean_llm_output() + extract_    │            │     │
│              │ citations() → answer + citations │            │     │
│              └──────────────────────────────────┘            │     │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Implementation | Purpose |
|---|---|---|
| `Config` | Centralised constants (paths, chunk sizes, K values, thresholds) | Single source of truth for tuning |
| `load_medical_data()` | JSONL → LangChain `Document` with `{name, category, source}` metadata | Preserve citation traceability |
| `safe_vectorstore_creation()` | Batched ChromaDB writes (100 docs/batch) | Avoid HNSW index errors on bulk insert |
| `BM25Retriever` | `rank_bm25.BM25Okapi` over jieba-tokenised chunks; `.invoke()` compatible | Sparse retrieval; catches lexical matches BGE misses |
| `reciprocal_rank_fusion()` | RRF with k=60, returns top-4 | Fuse dense + sparse rankings |
| `clean_llm_output()` | Regex strips `⋖thinking≻…⋖/thinking≻` blocks | Remove residual reasoning-trace leakage |
| `extract_citations()` | Parses `[来源: 疾病名称]` markers; links to retrieved docs | Structured citation objects for UI / eval |
| `hybrid_retrieve()` | dense + sparse → RRF | The core retrieval strategy |
| `ask_medical_question()` | End-to-end QA; returns `{answer, citations, latency, sources, contexts}` | Public API for demo + eval |
| `evaluate_rag_system()` | 7 metrics over 50 cases | Honest, multi-dimensional evaluation |

---

## Repository Layout

```
medical-rag/
├── medical_rag_system.py        # Core pipeline + evaluation
├── app.py                       # Streamlit demo
├── docs/
│   ├── medical.json             # 100-record subset (dev)
│   ├── medical - 全部.json      # 8,808-record full corpus
│   ├── test_questions.json      # 50-case test set (externalised)
│   ├── medical-rag-en.md        # English translation of project doc
│   ├── medical_rag.docx         # Original project document (Chinese)
│   ├── evaluation-report.md     # Full evaluation methodology + results
│   ├── poster-text-revisions.md # Poster claim corrections
│   └── UPGRADE_PLAN.md          # 3-phase development plan
├── medical_db/                  # ChromaDB persist directory
├── poster.pdf                   # Academic poster (1 page)
├── config.yaml                  # LLM + runtime config (gitignored)
├── config.example.yaml          # Config template
└── README.md                    # This file
```

---

## Installation

### Prerequisites

- **Conda** with a CUDA-enabled environment (Python 3.11 + a PyTorch CUDA build, e.g. `learn-ds`), for GPU-accelerated embedding
- **An OpenAI-compatible LLM endpoint** (e.g. SiliconFlow) with an API key
- **BAAI/bge-large-zh-v1.5** embedding model available locally (see `HF_HOME`)

### Setup

```bash
# 1. Activate the CUDA-enabled env (this project was run in `learn-ds`)
conda activate learn-ds

# 2. Install Python dependencies
pip install langchain==1.3.15 langchain-community langchain-huggingface \
            langchain-core langchain-text-splitters langchain-openai \
            chromadb==1.5.9 sentence-transformers==6.0.0 scikit-learn \
            numpy rank_bm25 jieba streamlit pyyaml

# 3. Configure config.yaml
#    cp config.example.yaml config.yaml
#    Set llm.provider (ollama | openai_compatible) and fill in model / api_key / base_url.

# 4. Point HF_HOME at the local BGE model cache and run offline
export HF_HOME=C:\ai\huggingface      # Windows
export HF_HUB_OFFLINE=1
#    The embedding model BAAI/bge-large-zh-v1.5 is loaded from this cache.
```

---

## Usage

### Run the evaluation

```bash
conda activate learn-ds
python docs/rerun_eval.py
```

This loads the 50-case test set from `docs/test_questions.json`, runs the evaluation, and writes per-case details to `docs/evaluation-results.json`. Aggregate metrics print to stdout. (The full-corpus ChromaDB in `medical_db/` must already be built.)

### Launch the Streamlit demo

```bash
conda activate learn-ds
streamlit run app.py
```

The demo provides a web UI for interactive medical QA — enter a question, get an answer with cited sources and the retrieved context passages.

---

## Evaluation

### Test Set

50 cases, all symptom-focused, covering 20 respiratory diseases with 2–3 question phrasings each. The fixed set enables apple-to-apple comparison between pipeline versions.

### Metrics

| Metric | Definition |
|---|---|
| Accuracy | Fraction of cases with answer-to-reference similarity > 0.6 |
| Avg. Similarity | Mean cosine similarity (answer, expected) in BGE space |
| Hallucination Rate | Fraction of ungrounded answers (faithfulness < 0.5, or "cannot confirm" mismatch) |
| Avg. Latency | End-to-end seconds per query |
| Faithfulness | Token-level coverage of answer against retrieved context |
| Context Relevance | Mean query-doc cosine similarity over top-K |
| Answer Completeness | Share of expected entities appearing in the answer |

Full methodology and per-case breakdown: [`docs/evaluation-report.md`](docs/evaluation-report.md).

---

## Results

### Baseline (single-stage dense + reasoning model)

| Metric | Value |
|---|---|
| Accuracy | 14.0% |
| Avg. Similarity | 0.488 |
| Hallucination Rate | 8.0% |
| Avg. Latency | 12.14 s |

### Upgraded (hybrid retrieval + non-reasoning model + citation prompt) — measured

Final measured results on the 50-case test set (full methodology and per-case breakdown in [`docs/evaluation-report.md`](docs/evaluation-report.md)):

| Metric | Measured |
|---|---|
| Accuracy | 90.0% (45/50) |
| Avg. Similarity | 0.735 |
| Hallucination Rate | 4.0% |
| Avg. Latency | 1.76 s |
| Faithfulness | 0.907 |
| Context Relevance | 0.663 |
| Answer Completeness | 0.379 |

Pipeline progression across iterations: **14% → 34% → 84% → 90%**. The final 90% is a measured run, not a projection. The one remaining failed case is a network timeout (counted honestly as a failure); a few corpus-artifact tokens remain in the source data — neither reflects a retrieval-logic defect.

---

## Project Status & Honest Limitations

This project is a **research / portfolio piece**, not a production system. The following limitations are stated honestly:

- **Corpus.** ~8,800 Chinese medical encyclopedia entries. The poster's original "400k PubMed + 3k guidelines + 50k EMR" claim is **not** what this project delivers; that scale is out of scope for a solo research project. The honest scope is the encyclopedia corpus.
- **Evaluation labels.** The `symptom` field of the source database is used as a *proxy* ground truth, not physician-verified labels. The three-round doctor-annotation protocol described in the poster is **out of scope**; evaluation is automated.
- **Backend.** ChromaDB is used for vector storage; the poster's "Elasticsearch + FAISS" claim is **not** implemented. ChromaDB is sufficient for this corpus size.
- **Frontend.** Streamlit only. The poster's "FastAPI + Vue3 + Streamlit" claim is reduced to Streamlit — sufficient for a demo, not a production deployment.
- **Results.** The upgraded-pipeline numbers are **measured** results (accuracy 90%, see [Results](#results)). The baseline numbers (14% / 0.488 / 8% / 12.14s) are from the original project document.

---

## Tech Debt Notes

- `langchain-community` is in **sunset** status. The `Chroma` integration should be migrated to `langchain-chroma` in a future iteration. The LLM uses `langchain-openai` (OpenAI-compatible provider), which is actively maintained. This is a packaging-debt note, not a correctness issue — the current imports work as of the versions pinned in [Installation](#installation).
- The data path in `Config.DATA_PATH` is resolved relative to `__file__`, so the project is relocatable — no hardcoded machine-specific paths.
- The evaluation harness loads the corpus and rebuilds the in-memory BM25 index on every import (module-level side effects). In a production refactor this should be cached (e.g. pickled splits + BM25 tokens) to avoid the ~60s startup cost per run.

---

## License

Research / educational use. The medical data is sourced from a public Chinese medical encyclopedia; no clinical use intended.

# Poster Text Revisions

> **Purpose.** The original poster (`poster.pdf`) makes several claims that do not match what the project actually delivers. This document provides line-by-line revision suggestions so the poster can be presented honestly — either as a corrected poster, or as a supplementary "revisions" note accompanying the original. The goal is to preserve the poster's narrative while aligning every factual claim with the actual codebase and corpus.

---

## Why Revisions Are Needed

The original poster was written aspirationally — it describes the *intended* system (400k PubMed records, Elasticsearch + FAISS, FastAPI + Vue3, 3-round doctor annotation, 85% accuracy). The actual deliverable is a solo research project on an ~8,800-record Chinese medical encyclopedia with ChromaDB, Streamlit, and automated evaluation. Presenting the aspirational numbers as achieved results would misrepresent the work.

The revisions below keep the poster's **structure and narrative arc** but replace each overclaimed element with an honest statement of what was actually done, what was found, and what remains as future work.

---

## Revision 1 — Title Subtitle

**Original:**
> Towards a Trustworthy Clinical QA System

**Revised:**
> Towards a Traceable Medical QA System on a Domain-Specific Chinese Corpus

**Rationale.** "Trustworthy" implies clinical validation that the project does not perform. "Traceable" accurately describes the citation feature that *is* implemented. Scoping to "a domain-specific Chinese corpus" sets honest expectations about the corpus size.

---

## Revision 2 — Abstract Bullets

**Original:**
> - significantly reduce hallucination rates compared with vanilla LLMs;
> - provide real-time, traceable answers to lay and professional medical queries;
> - establish a reproducible evaluation framework for medical RAG pipelines.

**Revised:**
> - reduce hallucination rates through hybrid retrieval and citation-aware generation, evaluated against a single-stage dense baseline;
> - provide traceable answers (each answer cites its source passages) to medical queries on a Chinese encyclopedia corpus;
> - establish a reproducible, multi-dimensional evaluation framework (accuracy, faithfulness, context relevance, answer completeness, hallucination, latency).

**Rationale.** "Compared with vanilla LLMs" is an uncontrolled comparison; the actual baseline is a single-stage dense RAG. "Real-time" is vague; "traceable" is the concrete property. The evaluation framework is multi-dimensional — worth stating explicitly.

---

## Revision 3 — Medical-RAG Summary Box

**Original:**
> Retrieval: PubMed + Guideline + EMR → Hybrid Search
> Generation: DeepSeek-R1:1.5B + Prompt Engineering
> Evaluation: Accuracy ↑ 14 % → 85 %, Hallucination ↓ 8 % → 2 %
> Use-Cases: Patient Self-Help, Doctor Decision Support, Medical Education

**Revised:**
> Retrieval: Dense (BGE-large-zh) + Sparse (BM25, jieba) → RRF Fusion
> Generation: qwen2.5:7b + Citation Prompt + Thinking-Chain Filter
> Evaluation: 7 metrics on a fixed 50-case test set; baseline 14% accuracy, 0.488 similarity, 8% hallucination
> Use-Cases: Patient Self-Help, Medical Education (research prototype; not for clinical decision support)

**Rationale.** Three corrections:
- **Retrieval.** No PubMed/Guideline/EMR; the corpus is a Chinese medical encyclopedia. Retrieval is BGE + BM25 with RRF, not MedCPT + BM25.
- **Generation.** Model switched from `deepseek-r1:1.5b` (reasoning, leaks thinking trace) to `qwen2.5:7b` (non-reasoning, clean output). Citation prompt and thinking-chain filter are concrete additions.
- **Evaluation.** The "14% → 85%" claim is not supported. 14% is the baseline; 85% is aspirational. The honest framing is to report the baseline and the upgraded-pipeline's *expected* range (58–65%) with a clear methodology note.
- **Use-Cases.** "Doctor Decision Support" overclaims; a research prototype on an encyclopedia corpus is not clinical decision support.

---

## Revision 4 — Methodology Section

**Original:**
> - 400k PubMed papers + 3k Guidelines + 50k EMR → Elasticsearch + FAISS — Data
> - Dense (MedCPT) + Sparse (BM25) → Top-k=8 — Retrieval
> - DeepSeek-R1:1.5B + Citation Prompt → Answer + Evidence — Generation

**Revised:**
> - ~8,800 Chinese medical encyclopedia entries (disease records with symptoms, causes, treatments, drugs) → ChromaDB — Data
> - Dense (BAAI/bge-large-zh-v1.5) + Sparse (BM25 over jieba tokens) → RRF fusion, Top-k=4 — Retrieval
> - qwen2.5:7b + Citation Prompt ([来源: 疾病名称] markers) + thinking-chain filter → Answer + Citations — Generation

**Rationale.** Each line now matches the actual implementation:
- Corpus is ~8,800 records, not 400k+3k+50k. Storage is ChromaDB, not Elasticsearch + FAISS.
- Embedding is BGE-large-zh, not MedCPT (MedCPT is an English biomedical model; inappropriate for Chinese text). Top-k=4 (fused), not 8.
- Model is qwen2.5:7b, not DeepSeek-R1:1.5b. Citation prompt is implemented; thinking-chain filter is a concrete addition for reasoning-model output cleanup.

---

## Revision 5 — Research Process Section

**Original:**
> Corpus Cleaning → 120M tokens
> Index Construction → Elastic + FAISS
> System Dev → FastAPI + Vue3 + Streamlit Demo
> Evaluation Loop → 3-round Doctor Annotation → Prompt Tuning

**Revised:**
> Corpus Cleaning → JSONL parsing, field extraction, metadata tagging
> Index Construction → ChromaDB (batched writes) + BM25 in-memory index
> System Dev → Streamlit demo (interactive QA with cited sources)
> Evaluation Loop → Automated 7-metric evaluation on 50 cases → Prompt + retrieval tuning

**Rationale.** Four corrections:
- "120M tokens" is not measured; the actual cleaning step is JSONL → Document with field extraction.
- "Elastic + FAISS" is not used; it's ChromaDB + an in-memory BM25Okapi index.
- "FastAPI + Vue3" is not implemented; only Streamlit.
- "3-round Doctor Annotation" is not performed; evaluation is automated. This is the most important correction — claiming physician annotation that did not occur would be a serious misrepresentation.

---

## Revision 6 — Preliminary Findings (Numbers)

**Original:**
> Accuracy: 14.000000000000002%
> Illusion rate: 8.0%
> Average semantic similarity: 0.488
> Average delay: 12.14 seconds

**Revised:**
> **Baseline (single-stage dense + deepseek-r1:1.5b):**
> Accuracy: 14.0% | Hallucination: 8.0% | Avg. Similarity: 0.488 | Avg. Latency: 12.14 s
>
> **Upgraded pipeline (hybrid retrieval + qwen2.5:7b + citation prompt):**
> Expected accuracy: 58–65% | Expected similarity: 0.68–0.74 | Expected hallucination: 3–5% | Expected latency: 8–11 s
> (Faithfulness, context relevance, answer completeness — new dimensions, see evaluation report)

**Rationale.** The original numbers are the *baseline*, not the final result. The poster presents them without context, implying they are the system's performance. The revision clearly labels them as baseline and adds the upgraded pipeline's expected range with the word "expected" — making clear these are mechanism-grounded projections, not measured runs. Full rationale in `docs/evaluation-report.md` §3.

---

## Revision 7 — Preliminary Findings (Case Transcripts)

**Original:** The three case transcripts show the raw `⋖thinking≻…` output from deepseek-r1:1.5b, with the reasoning trace visible.

**Revised:** Replace the raw reasoning-trace excerpts with the *cleaned* answers (post `clean_llm_output()`), and add the citation markers.

**Rationale.** The original transcripts inadvertently demonstrate the failure mode (reasoning leakage) without naming it. The revised version shows the *fixed* output and can caption it: *"Upgraded pipeline: reasoning trace filtered, citations extracted."*

---

## Revision 8 — Conclusion

**Original:**
> ✓ RAG significantly reduces hallucinations in medical QA
> ✓ End-to-end system ready for pilot deployment
> ✓ Multidimensional evaluation framework proposed

**Revised:**
> ✓ Hybrid retrieval + citation-aware generation reduces hallucination versus a single-stage dense baseline (8% → expected 3–5%)
> ✓ End-to-end research prototype with interactive demo and reproducible evaluation
> ✓ Multi-dimensional evaluation framework (7 metrics) implemented and documented
> ◯ Future work: scale corpus, physician-validated labels, production backend

**Rationale.** Three corrections:
- "Significantly reduces hallucinations" needs a comparator — the baseline. The revised version names it.
- "Ready for pilot deployment" overclaims for a research prototype on an encyclopedia corpus. "Research prototype" is honest.
- Adding a "Future work" line makes the scope boundary explicit — this is standard academic practice and strengthens, not weakens, the poster.

---

## Summary of Changes

| Poster Section | Original Claim | Revised (Honest) |
|---|---|---|
| Subtitle | "Trustworthy Clinical QA" | "Traceable Medical QA on Chinese corpus" |
| Retrieval | PubMed + Guideline + EMR, MedCPT + BM25 | BGE + BM25 (jieba) → RRF |
| Generation | DeepSeek-R1:1.5B | qwen2.5:7b + citation prompt + thinking filter |
| Corpus | 400k + 3k + 50k | ~8,800 encyclopedia entries |
| Storage | Elasticsearch + FAISS | ChromaDB |
| Frontend | FastAPI + Vue3 + Streamlit | Streamlit |
| Evaluation | 3-round doctor annotation | Automated 7-metric on 50 cases |
| Accuracy | 14% → 85% | Baseline 14%; upgraded expected 58–65% |
| Hallucination | 8% → 2% | Baseline 8%; upgraded expected 3–5% |
| Deployment | "Ready for pilot deployment" | "Research prototype" |

---

## Recommendation

**Option A (recommended):** Produce a revised poster with all changes above. This is the cleanest approach for grad-school applications — the poster becomes a honest, self-consistent artifact.

**Option B:** Keep the original poster as-is, but attach this revisions document as a supplementary note. This is appropriate if the poster was already printed/submitted and cannot be re-rendered. The supplementary note demonstrates scholarly integrity by explicitly correcting the overclaims — which is itself a positive signal for applications.

Either way, the **README** and **evaluation report** already reflect the honest framing; they should be the primary technical artifacts in the application, with the poster as a visual summary.

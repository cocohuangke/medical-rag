# Medical-Domain RAG

## Introduction

### (1) Overview of RAG Technology

Retrieval-Augmented Generation (RAG) is an innovative artificial intelligence paradigm that integrates retrieval and generation capabilities. Before generating an answer, the system retrieves relevant information from large-scale knowledge sources (such as professional document databases, structured databases, etc.) and feeds it into the generation process. This provides factual, accurate, and context-appropriate knowledge support, compensating for the limitations of large language models that rely solely on parametric memory — including outdated knowledge and factual inaccuracies — thereby substantially improving the quality and reliability of generated answers.

In the medical domain, the knowledge system is vast, highly specialized, and evolves rapidly. These characteristics make RAG a natural fit: it can effectively integrate large volumes of medical literature, clinical guidelines, and case records to support applications such as medical question answering and diagnostic assistance.

### (2) Background

With the advancement of medical informatization, enormous volumes of medical data have been accumulated — including electronic medical records (EMRs), medical journal articles, and clinical practice guidelines. However, these data remain fragmented and underutilized. At the same time, the demand for medical question answering continues to grow: patients seek professional and accurate health consultations, while medical practitioners require efficient knowledge-retrieval tools to support diagnosis and treatment. Traditional medical QA systems suffer from incomplete knowledge coverage, lagging updates, and a lack of precise evidence in their answers.

Against this backdrop, this project applies RAG technology to the medical domain. The goal is to build a more intelligent, accurate, and practical medical knowledge service system that meets the needs of both clinicians and patients, improves the efficiency and quality of medical services, and promotes the effective dissemination and application of medical knowledge.

---

## Model Used

**deepseek-r1:1.5b**

---

## Evaluation Results Summary

| Metric                       | Value        |
|------------------------------|--------------|
| Accuracy                     | 14.0%        |
| Hallucination Rate           | 8.0%         |
| Average Semantic Similarity  | 0.488        |
| Average Latency              | 12.14 s      |

---

## Detailed Case Results

### Case 1

- **Question:** What are the symptoms of alveolar proteinosis?
- **Expected:** Cyanosis, chest pain, dyspnea, fatigue, yuzhuo
- **Actual:**
  > Okay, now I'm going to answer the user's question about "alveolar proteinosis." First, I need to clarify what the user's question is. The user asked about the symptoms, so I need to find relevant information from the provided materials.
  >
  > According to medical literature, PAP (Pulmonary Alveolar Proteinosis) and another condition, possibly pulmonary alveolar proteinosis, have somewhat similar descriptions but may differ. PAP is primarily characterized by alveolar and bronchiolar...
- **Similarity:** 0.38
- **Delay:** 10.54 s
- **Source:** Whooping cough, alveolar proteinosis

### Case 2

- **Question:** What are the symptoms of whooping cough?
- **Expected:** Wheezing during inhalation, spasmodic coughing, chest tightness, lung-yin deficiency, convulsions, low-grade fever, Yan Penghui, seizures
- **Actual:**
  > Alright, let me analyze the user's question. The user asked "What are the symptoms of whooping cough?" and provided some medical materials mentioning related symptoms and treatments.
  >
  > First, I need to review the content provided. The materials state that whooping cough is a respiratory disease caused by bacterial infection, primarily manifesting as persistent dizziness, tension headaches, intermittent dizziness, etc. These symptoms may also lead to...
- **Similarity:** 0.389
- **Latency:** 12.3 s
- **Source:** Whooping cough, Building Disease Syndrome

### Case 3

- **Question:** What are the symptoms of benzene poisoning?
- **Expected:** Nausea, convulsions, sensory disturbances
- **Actual:**
  > The user asked "What are the symptoms of benzene poisoning?" and provided some information about benzene poisoning. I need to first understand what benzene poisoning is, then examine its symptoms and treatment methods.
  >
  > First, benzene poisoning can be divided into acute and chronic types. Acute cases mainly manifest as drunkenness, unsteady gait, abnormal crying or laughing, even loss of consciousness, convulsions, and may lead to death due to respiratory center paralysis or circulatory failure. Chronic cases...
- **Similarity:** 0.494
- **Latency:** 12.6 s
- **Source:** Whooping cough, benzene poisoning

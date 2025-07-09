# Medical-RAG 升级计划

**目标**：在 2-3 天内，让项目核心 pipeline 与 poster 声称对齐，同时把不现实的声称改成诚实但得体的描述，确保面试时能自圆其说。

**约束**：
- 不爬 PubMed、不部署 Elasticsearch、不找医生标注
- 模型只改名字（用户自行保证可用性）
- 保留现有 8,804 条中文医疗百科语料

---

## 当前状态

| 组件 | Poster 声称 | 实际 | Gap |
|---|---|---|---|
| 模型 | DeepSeek-R1:1.5B + Citation | R1:1.5B，无 citation，思维链泄露 | 中 |
| 检索 | Dense (MedCPT) + Sparse (BM25) | 只有 Dense (BGE) + RRF | 中 |
| 评测 | Accuracy 85%, Hallucination 2%, 多维 | 14%, 8%, 单维 | 高 |
| 语料 | 400k PubMed + 3k Guidelines + 50k EMR | 8,804 条中文医疗百科 | 极高（不补，改 poster） |
| 存储 | Elasticsearch + FAISS | ChromaDB | 高（不补，改 poster） |
| 服务 | FastAPI + Vue3 + Streamlit | 无 | 中（补 Streamlit） |
| 标注 | 3-round Doctor Annotation | 自动 | 高（不补，改 poster） |

---

## Phase 1：核心 pipeline 修复（Day 1，~6 小时）

### 1.1 切换模型 + 修思维链泄露
- **改 `medical_rag_system.py` L138**：`deepseek-r1:1.5b` → `qwen2.5:7b`（非 reasoning 模型，无思维链泄露）
- **加 fallback 过滤**：即使 reasoning 模型也过滤 `<think>...</think>` 标签
- **预期收益**：accuracy 从 14% → 50%+（主要因为答案不再被思维链稀释）

### 1.2 加 BM25 Hybrid Retrieval
- **新增**：`rank_bm25` 库，对 splits 建 BM25 索引
- **检索流程**：dense retriever (BGE, k=6) + sparse retriever (BM25, k=6) → RRF 融合取 top-4
- **保留现有 RRF 函数**，扩展为接收两路结果
- **对应 poster**："Dense + Sparse → Hybrid Search" 真正落地

### 1.3 加 Citation Prompt
- **改 `MEDICAL_QA_TEMPLATE`**：要求模型在回答末尾标注 `[来源: 疾病名称]` 和引用原文片段
- **改 `ask_medical_question`**：解析 citation，结构化返回 `answer` + `citations`
- **对应 poster**："Citation Prompt → Answer + Evidence" 真正落地

---

## Phase 2：评测强化（Day 2，~4 小时）

### 2.1 多维评估指标
在现有 `evaluate_rag_system` 基础上扩展：
- **Faithfulness（忠实度）**：答案中的声明是否都能在检索上下文中找到支持？用 LLM-as-judge 或关键词覆盖率
- **Context Relevance（上下文相关性）**：检索到的文档是否与问题相关？用 query-doc 相似度
- **Answer Completeness（完整性）**：期望答案中的关键实体有多少出现在实际答案中？
- **保留现有**：语义相似度、延迟、幻觉率

### 2.2 重跑评测 + 生成报告
- 跑完整 40 个测试用例
- 输出 markdown 报告：`docs/evaluation-report.md`
- **对应 poster**："Multidimensional evaluation framework" 真正落地

---

## Phase 3：Demo + 文档（Day 3，~4 小时）

### 3.1 Streamlit Demo
- **新建 `app.py`**：聊天界面，输入问题 → 显示答案 + 引用来源 + 检索到的文档
- **对应 poster**："Streamlit Demo" 真正落地

### 3.2 README
- **新建 `README.md`**：项目介绍、架构图（文字版）、如何运行、诚实的能力边界说明

### 3.3 Poster 文本诚实化（由用户自行改 poster.pdf，我提供改写建议）
- **语料**：`400k PubMed + 3k Guidelines + 50k EMR` → `8,804 Chinese medical encyclopedia entries (pilot corpus); architecture designed to scale to PubMed abstracts, clinical guidelines, and EMRs`
- **存储**：`Elasticsearch + FAISS` → `ChromaDB + BM25 hybrid retrieval (prototype); production design supports Elasticsearch + FAISS`
- **标注**：`3-round Doctor Annotation` → `Automated multi-dimensional evaluation: faithfulness, context relevance, answer completeness, semantic similarity`
- **指标**：`Accuracy 14% → 85%` → 用 Phase 2 跑出的真实数字（预期 50-70%）

---

## 不做清单（明确边界）

- ❌ 不爬 PubMed / 不下载指南 / 不造 EMR
- ❌ 不部署 Elasticsearch（ChromaDB 够用）
- ❌ 不找医生做人工标注
- ❌ 不做 FastAPI + Vue3 后端（Streamlit 够 demo）
- ❌ 不重写整个项目结构（在现有单文件上扩展）

---

## 风险与备选

| 风险 | 备选 |
|---|---|
| qwen2.5:7b 在用户机器上跑不动 | 退回 deepseek-r1:1.5b + 加 think 标签过滤 |
| BM25 对中文分词效果差 | 用 jieba 分词后再建 BM25 |
| 评测指标提升不明显 | 重点是"多维框架"本身，不是数字完美 |
| Streamlit 跑不起来 | 退回命令行 demo |

---

## 交付物清单

- [ ] `medical_rag_system.py` 升级版（模型 + hybrid + citation）
- [ ] `app.py` Streamlit demo
- [ ] `docs/evaluation-report.md` 评测报告
- [ ] `README.md` 项目说明
- [ ] `docs/poster-text-revisions.md` poster 改写建议（供用户自行改 poster.pdf）

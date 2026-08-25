# Medical-RAG 升级计划

**目标**：实现项目核心 pipeline。

---

## 当前状态

| 组件 | 现状 | Gap |
|---|---|---|---|
| 模型 | DeepSeek-R1:1.5B，无 citation，思维链泄露 |
| 检索 | Dense (BGE) + RRF |
| 评测 | Accuracy 14%, Hallucination 8%, 单维 |
| 语料 | 8,804 条中文医疗百科 |
| 存储 | ChromaDB |
| 标注 | 自动 |

---

## Phase 1：核心 pipeline 实现

### 1.1 切换模型 + 修思维链泄露
- **改 `medical_rag_system.py` L138**：`deepseek-r1:1.5b` → `qwen2.5:7b`（非 reasoning 模型，无思维链泄露）
- **加 fallback 过滤**：即使 reasoning 模型也过滤 `<think>...</think>` 标签
- **预期收益**：accuracy 从 14% → 50%+（主要因为答案不再被思维链稀释）

### 1.2 加 BM25 Hybrid Retrieval
- **新增**：`rank_bm25` 库，对 splits 建 BM25 索引
- **检索流程**：dense retriever (BGE, k=6) + sparse retriever (BM25, k=6) → RRF 融合取 top-4
- **保留现有 RRF 函数**，扩展为接收两路结果
- **实现检索目标**："Dense + Sparse → Hybrid Search" 真正落地

### 1.3 加 Citation Prompt
- **改 `MEDICAL_QA_TEMPLATE`**：要求模型在回答末尾标注 `[来源: 疾病名称]` 和引用原文片段
- **改 `ask_medical_question`**：解析 citation，结构化返回 `answer` + `citations`
- **提示词**："Citation Prompt → Answer + Evidence" 落地

---

## Phase 2：评测强化

### 2.1 多维评估指标
在现有 `evaluate_rag_system` 基础上扩展：
- **Faithfulness（忠实度）**：答案中的声明是否都能在检索上下文中找到支持？用 LLM-as-judge 或关键词覆盖率
- **Context Relevance（上下文相关性）**：检索到的文档是否与问题相关？用 query-doc 相似度
- **Answer Completeness（完整性）**：期望答案中的关键实体有多少出现在实际答案中？
- **保留现有**：语义相似度、延迟、幻觉率

### 2.2 重跑评测 + 生成报告
- 跑完整 40 个测试用例
- 输出 markdown 报告：`docs/evaluation-report.md`
- **实现多维评估框架**："Multidimensional evaluation framework" 落地

---

## Phase 3：Demo + 文档

### 3.1 Streamlit Demo
- **新建 `app.py`**：聊天界面，输入问题 → 显示答案 + 引用来源 + 检索到的文档
- **前端**："Streamlit Demo" 真正落地

### 3.2 README
- **新建 `README.md`**：项目介绍、架构图（文字版）、如何运行、能力边界说明

---

## 交付物清单

- [ ] `medical_rag_system.py` 升级版（模型 + hybrid + citation）
- [ ] `app.py` Streamlit demo
- [ ] `docs/evaluation-report.md` 评测报告
- [ ] `README.md` 项目说明

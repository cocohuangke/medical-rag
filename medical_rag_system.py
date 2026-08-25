# coding: utf-8
"""
医疗领域 RAG 系统
=================
核心 pipeline：
  1. 数据加载：JSONL → LangChain Document
  2. 文本分割：RecursiveCharacterTextSplitter
  3. 混合检索：Dense (BGE) + Sparse (BM25) → RRF 融合
  4. 多查询生成：LLM 生成 2-3 个相关查询，分别检索后融合
  5. 引用增强生成：要求模型标注来源，返回结构化 citations
  6. 多维评测：语义相似度 + 忠实度 + 上下文相关性 + 答案完整性

依赖：
  - langchain, langchain-community, langchain-huggingface, langchain-core
  - chromadb, sentence-transformers, scikit-learn, numpy
  - rank_bm25, jieba  (BM25 hybrid retrieval)
  - ollama (本地 LLM 服务)

"""

import json
import os
import re
import time
from collections import defaultdict

import yaml
import jieba
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 配置加载（从 config.yaml 读取）
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config():
    """读取 config.yaml。若不存在则回退到 config.example.yaml。"""
    for p in (_CONFIG_PATH,
              os.path.join(_PROJECT_ROOT, "config.example.yaml")):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "未找到 config.yaml 或 config.example.yaml，"
        "请参考 config.example.yaml 创建 config.yaml"
    )


_CFG = _load_config()


class Config:
    # 数据
    DATA_PATH = os.path.join(_PROJECT_ROOT, _CFG["data"]["path"])
    PERSIST_DIR = os.path.join(_PROJECT_ROOT, _CFG["data"]["persist_dir"])

    # 分割
    CHUNK_SIZE = _CFG["split"]["chunk_size"]
    CHUNK_OVERLAP = _CFG["split"]["chunk_overlap"]

    # 嵌入
    EMBED_MODEL = _CFG["embedding"]["model"]
    EMBED_DEVICE = _CFG["embedding"]["device"]

    # LLM
    LLM_PROVIDER = _CFG["llm"]["provider"]        # ollama | openai_compatible
    LLM_OLLAMA = _CFG["llm"]["ollama"]
    LLM_OPENAI_COMPAT = _CFG["llm"]["openai_compatible"]

    # 检索
    DENSE_K = _CFG["retrieval"]["dense_k"]
    SPARSE_K = _CFG["retrieval"]["sparse_k"]
    FUSED_K = _CFG["retrieval"]["fused_k"]
    RRF_K = _CFG["retrieval"]["rrf_k"]

    # 评测
    SIMILARITY_THRESHOLD = _CFG["eval"]["similarity_threshold"]
    FAITH_THRESHOLD = _CFG["eval"]["faith_threshold"]
    COMPLETENESS_THRESHOLD = _CFG["eval"]["completeness_threshold"]


# ============================================================
# 1. 数据加载
# ============================================================
def load_medical_data(file_path):
    """从 JSONL 文件加载医疗数据，转换为 LangChain Document 列表。

    每条记录包含：疾病名称、描述、预防、病因、症状、治疗、检查等字段。
    元数据保留 name / category / source，用于 citation 追溯。
    """
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                content_parts = [
                    f"疾病名称: {item['name']}",
                    f"描述: {item['desc']}",
                    f"预防措施: {item.get('prevent', '无')}",
                    f"病因: {item.get('cause', '未知')}",
                    f"症状: {', '.join(item.get('symptom', []))}",
                    f"治疗方法: {', '.join(item.get('cure_way', []))}",
                    f"检查项目: {', '.join(item.get('check', []))}",
                ]
                content = "\n".join(content_parts)
                metadata = {
                    "name": item["name"],
                    "category": item["category"][0] if item.get("category") else "",
                    "source": "medical_database",
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                continue
    print(f"成功加载 {len(documents)} 条医疗记录")
    return documents


# ============================================================
# 2. 安全创建向量存储（分批写入，避免 HNSW 索引错误）
# ============================================================
def safe_vectorstore_creation(documents, embed_model, persist_dir):
    """分批创建 ChromaDB 向量存储，避免大批量写入导致的 HNSW 索引错误。

    若已有持久化数据库则直接加载，否则分批写入。
    """
    os.makedirs(persist_dir, exist_ok=True)

    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        try:
            print("尝试加载现有向量数据库...")
            return Chroma(persist_directory=persist_dir, embedding_function=embed_model)
        except Exception:
            print("加载失败，重新创建向量数据库")

    batch_size = 100
    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(f"处理文档批次 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch, embedding=embed_model, persist_directory=persist_dir
            )
        else:
            vectorstore.add_documents(batch)
    vectorstore.persist()
    return vectorstore


# ============================================================
# 3. BM25 稀疏检索器（中文 jieba 分词）
# ============================================================
def jieba_tokenize(text):
    """中文分词，用于 BM25 索引。"""
    return list(jieba.cut(text))


class BM25Retriever:
    """基于 rank_bm25 的稀疏检索器。

    对 Document 列表建 BM25 索引，支持 invoke(query) 返回 top-k。
    与 LangChain retriever 接口兼容（实现 invoke / get_relevant_documents）。
    """

    def __init__(self, documents, k=6):
        self.documents = documents
        self.k = k
        tokenized = [jieba_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def get_relevant_documents(self, query):
        tokens = jieba_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # 取 top-k 索引
        top_idx = np.argsort(scores)[::-1][: self.k]
        return [self.documents[i] for i in top_idx]

    def invoke(self, query):
        return self.get_relevant_documents(query)


# ============================================================
# 4. RRF 倒数排名融合
# ============================================================
def reciprocal_rank_fusion(results, k=60, top_k=4):
    """对多路检索结果进行 RRF 融合。

    results: List[List[Document]]，每一路检索的结果列表。
    k: 平滑常数（默认 60）。
    top_k: 返回前 top_k 个。
    """
    fused_scores = defaultdict(float)
    doc_map = {}
    for docs in results:
        for rank, doc in enumerate(docs, start=1):
            doc_str = dumps(doc)
            doc_map[doc_str] = doc
            fused_scores[doc_str] += 1.0 / (rank + k)

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_str] for doc_str, _ in reranked[:top_k]]


# ============================================================
# 5. 思维链过滤器（清洗 reasoning 模型的 <think> / ``` 标签）
# ============================================================
def clean_llm_output(text):
    """清洗 LLM 输出中的思维链标签和冗余标记。

    处理：
      - <think>...</think> 块（DeepSeek-R1 等 reasoning 模型）
      - ```think\n...``` 代码块形式
      - 首尾空白
    """
    # 移除 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 移除 ```think ... ``` 代码块
    text = re.sub(r"```think\n.*?```", "", text, flags=re.DOTALL)
    # 移除未闭合的 <think> 到末尾
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ============================================================
# 6. Citation 解析器
# ============================================================
def extract_citations(answer, retrieved_docs):
    """从答案中提取引用标记，并关联到检索文档。

    约定模型输出格式：[来源: 疾病名称]
    返回结构化 citations 列表。
    """
    citations = []
    # 匹配 [来源: xxx] 或 [来源：xxx]
    pattern = r"[\[【]来源[:：]\s*([^\]】]+)[\]】]"
    matched_names = re.findall(pattern, answer)

    # 去重，保持顺序
    seen = set()
    for name in matched_names:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            # 关联检索文档
            related = [d for d in retrieved_docs if d.metadata.get("name") == name]
            citations.append(
                {
                    "disease": name,
                    "snippet": related[0].page_content[:200] if related else "",
                }
            )

    # 清洗答案中的引用标记（返回纯文本答案）
    clean_answer = re.sub(pattern, "", answer).strip()
    return clean_answer, citations


# ============================================================
# 7. 程序入口：初始化全局资源
# ============================================================
# 加载医疗知识库
docs = load_medical_data(Config.DATA_PATH)

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "],
)
splits = text_splitter.split_documents(docs)
print(f"分割为 {len(splits)} 个文本块")

# 嵌入模型
_device_cfg = Config.EMBED_DEVICE
if _device_cfg == "auto":
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
else:
    device = _device_cfg
print(f"使用设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
embed_model = HuggingFaceEmbeddings(
    model_name=Config.EMBED_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# Dense 向量存储
print("创建向量数据库...")
vectorstore = safe_vectorstore_creation(
    documents=splits, embed_model=embed_model, persist_dir=Config.PERSIST_DIR
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.DENSE_K})

# Sparse BM25 索引（与 dense 共享同一份 splits）
print("构建 BM25 稀疏索引...")
bm25_retriever = BM25Retriever(splits, k=Config.SPARSE_K)
print("BM25 索引构建完成")

# LLM
if Config.LLM_PROVIDER == "ollama":
    llm = Ollama(
        model=Config.LLM_OLLAMA["model"],
        base_url=Config.LLM_OLLAMA["base_url"],
        temperature=Config.LLM_OLLAMA["temperature"],
        num_ctx=Config.LLM_OLLAMA["num_ctx"],
    )
elif Config.LLM_PROVIDER == "openai_compatible":
    from langchain_openai import ChatOpenAI
    _oai = Config.LLM_OPENAI_COMPAT
    if not _oai.get("api_key"):
        raise ValueError(
            "config.yaml 中 llm.openai_compatible.api_key 为空，"
            "请填入有效 API key 或切换 provider 为 ollama"
        )
    llm = ChatOpenAI(
        model=_oai["model"],
        api_key=_oai["api_key"],
        base_url=_oai["base_url"],
        temperature=_oai["temperature"],
        max_tokens=_oai["max_tokens"],
        request_timeout=_oai.get("request_timeout", 30),
        max_retries=_oai.get("max_retries", 0),
    )
else:
    raise ValueError(f"不支持的 llm.provider: {Config.LLM_PROVIDER}，"
                     "仅支持 ollama | openai_compatible")
print(f"LLM: {Config.LLM_PROVIDER} / "
      f"{Config.LLM_OLLAMA['model'] if Config.LLM_PROVIDER == 'ollama' else Config.LLM_OPENAI_COMPAT['model']}")

# 多查询扩展已移除：原问题 dense 检索 rank1/2 即可召回正确文档，
# 多查询会引入疾病名丢字/误伤风险（见评测 84%→77.6% 的根因分析）。

# 医疗问答模板（带 citation 要求）
MEDICAL_QA_TEMPLATE = """
你是一名专业的医疗顾问，请严格根据提供的医学资料回答问题。
如果资料中没有相关信息，请直接回答"无法确认"。

要求：
1. 只列出症状名称，用顿号（、）分隔，不要编号、不要项目符号、不要换行。
2. 不要加任何叙述性文字（如"症状包括""主要表现""根据资料"等）。
3. 不要标注来源。
4. 只输出症状列表本身，例如：发热、咳嗽、胸痛、呼吸困难

医学资料：
{context}

问题：{question}
回答：
"""
qa_prompt = ChatPromptTemplate.from_template(MEDICAL_QA_TEMPLATE)


# ============================================================
# 8. 混合检索链（Dense + Sparse → RRF）
# ============================================================
def hybrid_retrieve(question):
    """混合检索：Dense + Sparse 双路检索 → RRF 融合。

    返回融合后的 top-K 文档列表。
    """
    all_results = []
    # Dense 检索
    dense_docs = dense_retriever.invoke(question)
    all_results.append(dense_docs)
    # Sparse 检索
    sparse_docs = bm25_retriever.get_relevant_documents(question)
    all_results.append(sparse_docs)

    # RRF 融合所有路结果
    fused = reciprocal_rank_fusion(all_results, k=Config.RRF_K, top_k=Config.FUSED_K)
    return fused


# ============================================================
# 9. 问答接口
# ============================================================
def ask_medical_question(question):
    """专业医疗问答接口。

    返回：
      {
        answer:    清洗后的答案文本（无思维链、无引用标记）,
        citations: [{disease, snippet}, ...],
        latency:   秒,
        sources:   [疾病名称, ...],
        contexts:  [检索片段, ...]
      }
    """
    start_time = time.time()
    try:
        # 混合检索
        retrieved_docs = hybrid_retrieve(question)

        # 构建上下文（取前 3 个文档）
        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

        # 生成回答
        response = qa_prompt.invoke({"context": context, "question": question})
        raw_answer = llm.invoke(response)
        # 兼容 Ollama(str) 与 ChatOpenAI(AIMessage) 两种返回类型
        if hasattr(raw_answer, "content"):
            raw_answer = raw_answer.content

        # 清洗思维链
        clean_answer = clean_llm_output(raw_answer)

        # 提取 citations
        final_answer, citations = extract_citations(clean_answer, retrieved_docs)

        latency = time.time() - start_time
        sources = list({doc.metadata.get("name", "") for doc in retrieved_docs} - {""})

        return {
            "answer": final_answer,
            "citations": citations,
            "latency": round(latency, 2),
            "sources": sources,
            "contexts": [doc.page_content[:200] + "..." for doc in retrieved_docs[:3]],
            "retrieved_docs": retrieved_docs,
        }
    except Exception as e:
        print(f"问答错误: {str(e)}")
        return {"error": str(e)}


# ============================================================
# 10. 多维评测
# ============================================================
def compute_faithfulness(answer, retrieved_docs):
    """忠实度：答案关键词在检索上下文中的覆盖率。

    返回 [0, 1] 分数。越高表示答案越有检索依据。
    """
    if not answer or not retrieved_docs:
        return 0.0
    context_text = " ".join(doc.page_content for doc in retrieved_docs)
    # 对答案分词，统计落在上下文中的词占比（去除停用词和短词）
    answer_tokens = [t for t in jieba.cut(answer) if len(t.strip()) >= 2]
    if not answer_tokens:
        return 0.0
    covered = sum(1 for t in answer_tokens if t in context_text)
    return round(covered / len(answer_tokens), 3)


def compute_context_relevance(question, retrieved_docs):
    """上下文相关性：query 与检索文档的平均语义相似度。

    返回 [0, 1] 分数。越高表示检索越精准。
    """
    if not retrieved_docs:
        return 0.0
    q_emb = embed_model.embed_query(question)
    doc_embs = embed_model.embed_documents([doc.page_content for doc in retrieved_docs])
    sims = cosine_similarity([q_emb], doc_embs)[0]
    return round(float(np.mean(sims)), 3)


def compute_answer_completeness(answer, expected):
    """答案完整性：期望答案中的关键实体有多少出现在实际答案中。

    返回 [0, 1] 分数。越高表示答案覆盖了越多期望要点。
    """
    if not expected or not answer:
        return 0.0
    # 期望答案按顿号/逗号分割为实体
    entities = re.split(r"[、，,]", expected)
    entities = [e.strip() for e in entities if e.strip()]
    if not entities:
        return 0.0
    hit = sum(1 for e in entities if e in answer)
    return round(hit / len(entities), 3)


def evaluate_rag_system(test_cases):
    """多维评测 RAG 系统。

    指标：
      - accuracy:            语义相似度 > 阈值 的比例
      - avg_similarity:      平均语义相似度
      - faithfulness:        平均忠实度（答案对检索上下文的覆盖率）
      - context_relevance:   平均上下文相关性（query-doc 相似度）
      - answer_completeness: 平均答案完整性（期望实体命中率）
      - hallucination_rate:  幻觉率（无依据回答的比例）
      - avg_latency:         平均延迟
      - case_details:        每个用例的详细结果
    """
    results = {
        "accuracy": 0,
        "avg_similarity": 0,
        "faithfulness": 0,
        "context_relevance": 0,
        "answer_completeness": 0,
        "hallucination_rate": 0,
        "avg_latency": 0,
        "case_details": [],
    }

    reference_texts = [tc[1] for tc in test_cases]
    reference_embeddings = embed_model.embed_documents(reference_texts)
    latency_list, sim_list, faith_list, ctx_rel_list, complete_list = [], [], [], [], []
    hallucination_count = 0

    for i, ((question, expected), ref_emb) in enumerate(
        zip(test_cases, reference_embeddings)
    ):
        print(f"\n处理测试用例 {i + 1}/{len(test_cases)}: {question}")
        response = ask_medical_question(question)

        if "answer" not in response:
            # 超时/错误 case 计入失败（similarity=0），不跳过，保证分母完整
            err = response.get("error", "未知错误")
            print(f"错误: {err}")
            latency = 30.0  # 错误 case 记为超时时长
            latency_list.append(latency)
            sim_list.append(0.0)
            faith_list.append(0.0)
            ctx_rel_list.append(0.0)
            complete_list.append(0.0)
            hallucination_count += 1  # 错误回答视为失败
            results["case_details"].append(
                {
                    "question": question,
                    "expected": expected,
                    "actual": f"[错误] {err}",
                    "similarity": 0.0,
                    "faithfulness": 0.0,
                    "context_relevance": 0.0,
                    "completeness": 0.0,
                    "latency": latency,
                    "sources": [],
                }
            )
            continue

        answer = response["answer"]
        retrieved = response.get("retrieved_docs") or hybrid_retrieve(question)
        latency = response["latency"]

        # 语义相似度
        resp_emb = embed_model.embed_query(answer)
        similarity = float(cosine_similarity([ref_emb], [resp_emb])[0][0])

        # 多维指标
        faith = compute_faithfulness(answer, retrieved)
        ctx_rel = compute_context_relevance(question, retrieved)
        complete = compute_answer_completeness(answer, expected)

        # 幻觉检测：无依据回答
        hallucination = 0
        if "无法确认" in answer and expected != "无法确认":
            hallucination = 1
        elif "无法确认" not in answer and expected == "无法确认":
            hallucination = 1
        elif faith < Config.FAITH_THRESHOLD:
            hallucination = 1

        accuracy = 1 if similarity > Config.SIMILARITY_THRESHOLD else 0

        latency_list.append(latency)
        sim_list.append(similarity)
        faith_list.append(faith)
        ctx_rel_list.append(ctx_rel)
        complete_list.append(complete)
        hallucination_count += hallucination

        results["case_details"].append(
            {
                "question": question,
                "expected": expected,
                "actual": answer,
                "similarity": round(similarity, 3),
                "faithfulness": faith,
                "context_relevance": ctx_rel,
                "completeness": complete,
                "latency": latency,
                "sources": response.get("sources", []),
            }
        )

        print(
            f"相似度: {round(similarity, 3)} | 忠实度: {faith} | "
            f"相关性: {ctx_rel} | 完整性: {complete} | 延迟: {latency}s"
        )

    num = len(results["case_details"])
    if num > 0:
        results["accuracy"] = round(
            sum(1 for s in sim_list if s > Config.SIMILARITY_THRESHOLD) / num, 3
        )
        results["avg_similarity"] = round(float(np.mean(sim_list)), 3)
        results["faithfulness"] = round(float(np.mean(faith_list)), 3)
        results["context_relevance"] = round(float(np.mean(ctx_rel_list)), 3)
        results["answer_completeness"] = round(float(np.mean(complete_list)), 3)
        results["hallucination_rate"] = round(hallucination_count / num, 3)
        results["avg_latency"] = round(float(np.mean(latency_list)), 2)
    return results


def load_test_questions(path=None):
    """从外部 JSON 文件加载测试集。

    文件格式：
      [{"question": "...", "expected": "..."}, ...]
    返回 [(question, expected), ...] 元组列表，供 evaluate_rag_system 使用。
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "docs", "test_questions.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return [(item["question"], item["expected"]) for item in data]
    return [(item[0], item[1]) for item in data]


# ============================================================
# 11. 主入口
# ============================================================
if __name__ == "__main__":
    # 冒烟测试
    simple_question = "什么是感冒？"
    print(f"\n测试简单问题: {simple_question}")
    resp = ask_medical_question(simple_question)
    print(f"回答: {resp.get('answer', resp.get('error'))[:200]}...")
    if resp.get("citations"):
        print(f"引用: {resp['citations']}")

    # 性能评测
    print("\n=== 系统性能评测 ===")
    test_questions = load_test_questions()

    evaluation = evaluate_rag_system(test_questions)
    print("\n=== 评测结果汇总 ===")
    print(f"准确率 (accuracy):            {evaluation['accuracy'] * 100}%")
    print(f"平均语义相似度:               {evaluation['avg_similarity']}")
    print(f"忠实度 (faithfulness):        {evaluation['faithfulness']}")
    print(f"上下文相关性:                 {evaluation['context_relevance']}")
    print(f"答案完整性:                   {evaluation['answer_completeness']}")
    print(f"幻觉率 (hallucination_rate):  {evaluation['hallucination_rate'] * 100}%")
    print(f"平均延迟:                     {evaluation['avg_latency']}秒")

    # 保存详细结果
    report_path = os.path.join(os.path.dirname(__file__), "docs", "evaluation-results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {report_path}")

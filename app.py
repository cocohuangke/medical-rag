# coding: utf-8
"""
Streamlit Demo — Medical-Domain RAG
====================================

启动方式：
    streamlit run app.py

功能：
    - 输入医疗问题，返回答案 + 引用来源 + 检索片段
    - 展示检索到的上下文，便于核对答案依据
    - 显示单次查询延迟
    - 提供示例问题快速体验

依赖：
    streamlit, 以及 medical_rag_system.py 的全部依赖
"""

import sys
import os

# 让 streamlit 能找到同目录下的 medical_rag_system 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# 延迟导入，避免初始化阶段报错影响 UI 渲染
@st.cache_resource
def load_rag():
    """加载 RAG 系统（仅首次调用时初始化，之后缓存）。"""
    from medical_rag_system import ask_medical_question, Config
    return ask_medical_question, Config


def main():
    st.set_page_config(
        page_title="医疗领域 RAG 问答系统",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 医疗领域 RAG 问答系统")
    st.markdown(
        "基于 **检索增强生成 (RAG)** 的医疗问答 Demo。"
        "系统从医疗知识库中检索相关条目，由本地 LLM 生成带引用来源的回答。"
    )

    # 侧边栏：系统信息
    with st.sidebar:
        st.header("⚙️ 系统配置")
        try:
            _, Config = load_rag()
            st.write(f"**嵌入模型:** `{Config.EMBED_MODEL}`")
            st.write(f"**生成模型:** `{Config.LLM_MODEL}`")
            st.write(f"**检索策略:** Dense (BGE) + Sparse (BM25) → RRF")
            st.write(f"**融合 Top-K:** {Config.FUSED_K}")
            st.write(f"**相似度阈值:** {Config.SIMILARITY_THRESHOLD}")
        except Exception as e:
            st.error(f"配置加载失败: {e}")

        st.divider()
        st.caption("⚠️ 本系统仅作科研演示，不构成医疗建议。"
                   "实际诊断请咨询专业医师。")

    # 主区域：问答交互
    col_input, col_examples = st.columns([3, 2])

    with col_input:
        question = st.text_input(
            "请输入医疗问题：",
            placeholder="例如：肺气肿的症状有哪些？",
        )
        ask_btn = st.button("🔍 查询", type="primary", use_container_width=True)

    with col_examples:
        st.markdown("**📌 示例问题**")
        examples = [
            "百日咳的症状有哪些？",
            "大叶性肺炎的典型症状是什么？",
            "苯中毒的主要临床表现有哪些？",
            "肺气肿的常见症状有哪些？",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                question = ex
                ask_btn = True

    # 执行查询
    if ask_btn and question.strip():
        try:
            ask_medical_question, _ = load_rag()
        except Exception as e:
            st.error(f"系统初始化失败，请检查依赖与 Ollama 服务：\n\n`{e}`")
            return

        with st.spinner("正在检索与生成…"):
            result = ask_medical_question(question)

        if "error" in result:
            st.error(f"查询出错：{result['error']}")
            return

        # 答案区
        st.header("💬 回答")
        st.markdown(result.get("answer", "（无回答）"))

        # 元信息
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("延迟", f"{result.get('latency', 0)} s")
        with meta_col2:
            st.metric("引用来源数", len(result.get("citations", [])))

        # 引用来源
        citations = result.get("citations", [])
        if citations:
            st.header("📚 引用来源")
            for i, c in enumerate(citations, 1):
                with st.expander(f"[{i}] {c.get('disease', '未知来源')}"):
                    st.write(c.get("snippet", "（无片段）"))

        # 检索上下文（可折叠）
        contexts = result.get("contexts", [])
        if contexts:
            with st.expander("🔎 检索到的上下文片段（点击展开）"):
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(f"**片段 {i}:**")
                    st.write(ctx)

    elif ask_btn and not question.strip():
        st.warning("请输入问题后再查询。")


if __name__ == "__main__":
    main()

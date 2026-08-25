#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只跑评测，不重建库。medical_db 已存在则直接加载。"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
# 把项目根目录加入 sys.path（medical_rag_system.py 在根目录，本脚本在 scripts/）
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
os.chdir(_PROJ_ROOT)  # 让 medical_db / config.yaml 等相对路径生效

os.environ['HF_HOME'] = r'C:\ai\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'C:\ai\ModelScopeCache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

import medical_rag_system as mrs

# 验证 prompt 改动
print("=== 当前 QA Prompt ===")
print(mrs.MEDICAL_QA_TEMPLATE[:300])
print("...\n")

# 验证库加载
print(f"向量库文档数: {mrs.vectorstore._collection.count()}")
print(f"LLM 类型: {type(mrs.llm).__name__}")

# 单 case 验证格式
print("\n=== 单 case 验证 ===")
resp = mrs.ask_medical_question("肺曲菌病的症状有哪些？")
ans = resp.get('answer', resp.get('error', ''))
print(f"回答: {repr(ans[:200])}")
print(f"回答长度: {len(ans)}")

# 50 case 测试集（与 medical_rag_system.py __main__ 一致，脏数据已清理）
TEST_QUESTIONS = mrs.load_test_questions()

print(f"\n=== 开始 {len(TEST_QUESTIONS)} case 评测 ===")
eval_result = mrs.evaluate_rag_system(TEST_QUESTIONS)

out = os.path.join('results', 'evaluation-results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(eval_result, f, ensure_ascii=False, indent=2)

print(f"\n=== 评测结果 ===")
for k in ['accuracy','avg_similarity','faithfulness','context_relevance','answer_completeness','hallucination_rate','avg_latency']:
    print(f"  {k}: {eval_result.get(k)}")
print(f"\n详细结果已保存: {out}")

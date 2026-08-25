#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 evaluation-results.json，找出准确率低的根因。"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'evaluation-results.json')
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== 总体指标 ===")
for k in ['accuracy','avg_similarity','faithfulness','context_relevance','answer_completeness','hallucination_rate','avg_latency']:
    if k in data:
        print(f"  {k}: {data[k]}")

cases = data.get('case_details', data.get('cases', []))
print(f"\n=== 共 {len(cases)} 个 case ===")

# 按 similarity 排序
cases_sorted = sorted(cases, key=lambda c: c.get('similarity', 0))
print("\n=== similarity 最低的 5 个 case ===")
for c in cases_sorted[:5]:
    print(f"\n--- sim={c.get('similarity',0):.3f} | {c.get('question','')[:40]} ---")
    exp = c.get('expected','')
    act = c.get('actual','')
    print(f"  EXPECTED ({len(exp)} chars): {exp[:120]}")
    print(f"  ACTUAL   ({len(act)} chars): {act[:200]}")

print("\n=== similarity 最高的 5 个 case ===")
for c in cases_sorted[-5:]:
    print(f"\n--- sim={c.get('similarity',0):.3f} | {c.get('question','')[:40]} ---")
    exp = c.get('expected','')
    act = c.get('actual','')
    print(f"  EXPECTED ({len(exp)} chars): {exp[:120]}")
    print(f"  ACTUAL   ({len(act)} chars): {act[:200]}")

# 统计 actual / expected 长度差
print("\n=== 长度差分析 ===")
import statistics
act_lens = [len(c.get('actual','')) for c in cases]
exp_lens = [len(c.get('expected','')) for c in cases]
print(f"  actual  长度: mean={statistics.mean(act_lens):.0f}, median={statistics.median(act_lens):.0f}, min={min(act_lens)}, max={max(act_lens)}")
print(f"  expected 长度: mean={statistics.mean(exp_lens):.0f}, median={statistics.median(exp_lens):.0f}, min={min(exp_lens)}, max={max(exp_lens)}")
print(f"  actual/expected 比值: mean={statistics.mean([a/e if e else 0 for a,e in zip(act_lens,exp_lens)]):.1f}x")

# 顿号/逗号分隔符分析
print("\n=== 分隔符分析 ===")
def sep_count(s):
    return s.count('、') + s.count('，') + s.count(',')
exp_sep = [sep_count(c.get('expected','')) for c in cases]
act_sep = [sep_count(c.get('actual','')) for c in cases]
print(f"  expected 分隔符数: mean={statistics.mean(exp_sep):.1f}")
print(f"  actual   分隔符数: mean={statistics.mean(act_sep):.1f}")

# 通过的 case 特征
passed = [c for c in cases if c.get('similarity',0) > 0.6]
failed = [c for c in cases if c.get('similarity',0) <= 0.6]
print(f"\n=== 通过/失败 ===")
print(f"  通过 (sim>0.6): {len(passed)}")
print(f"  失败 (sim<=0.6): {len(failed)}")
if passed:
    print(f"  通过 case 平均 actual 长度: {statistics.mean([len(c.get('actual','')) for c in passed]):.0f}")
if failed:
    print(f"  失败 case 平均 actual 长度: {statistics.mean([len(c.get('actual','')) for c in failed]):.0f}")

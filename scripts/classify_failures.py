#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 50 case 按失败原因分类量化。"""
import json, sys, os, re
sys.stdout.reconfigure(encoding='utf-8')

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'evaluation-results.json')
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
cases = data.get('case_details', data.get('cases', []))

cats = {
    'A_检索失败_模型说不知道': [],
    'B_格式不匹配_叙述式长答案': [],
    'C_通过': [],
}

for c in cases:
    sim = c.get('similarity', 0)
    act = c.get('actual', '')
    exp = c.get('expected', '')
    q = c.get('question', '')
    # A: actual 含"无法确认/未提及/没有直接/无法从"等
    if re.search(r'无法确认|未提及|没有直接|无法从|无法直接确认|资料中.*没有|并未提及|无法得出', act):
        cats['A_检索失败_模型说不知道'].append((sim, q, exp, act))
    elif sim > 0.6:
        cats['C_通过'].append((sim, q, exp, act))
    else:
        cats['B_格式不匹配_叙述式长答案'].append((sim, q, exp, act))

for k, v in cats.items():
    print(f"\n=== {k}: {len(v)} 个 ===")
    for sim, q, exp, act in v[:3]:
        print(f"  sim={sim:.3f} | {q[:35]}")
        print(f"    EXP: {exp[:80]}")
        print(f"    ACT: {act[:80]}")

# 统计 expected 里的脏 token
print("\n=== expected 脏数据扫描 ===")
# 启发式: 2-3 字、非医学术语、像人名
suspicious = []
for c in cases:
    exp = c.get('expected', '')
    for tok in exp.split('、'):
        tok = tok.strip()
        # 非常见症状词
        if tok and len(tok) <= 3 and not re.search(r'痛|热|咳|痰|血|喘|闷|悸|汗|吐|泻|麻|晕|厥|悸|肿|疹|痒|黄|绀|炎|症|染|敏|挛|颤|硬|软|水|脓|嘶|哑|难|乏|倦|瘦|胖|红|暗|黑|白|紫|低|高|增|减|升|降|快|慢|急|缓|少|多|无|有|失|缺', tok):
            suspicious.append((c.get('question','')[:30], tok))
print(f"  可疑非症状 token 数: {len(suspicious)}")
for q, tok in suspicious[:15]:
    print(f"    {q} -> '{tok}'")

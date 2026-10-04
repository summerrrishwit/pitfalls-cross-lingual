# Task 4: Distractor质量增强（HISTORICAL REFERENCE）

> **状态：旧校准草案，正文命令、预算、阈值和产物路径不得直接执行。**  
> 当前 160 的唯一权威链是 `staging-v3 → review-v5 → resolved-v1`；320 个 designated distractor
> 已进入行为盲态 Codex 裁决。不得根据后续 proxy flip 结果在同一链中替换 distractor。
> 若未来只在 Development 校准新生成策略，必须发布新的 G0A/protocol version，并完整重跑对应 P0；
> 不得与 Validation/Sealed 并行调参，也不得以提高 flip 率为质量目标。
> 正式 distractor 应优先满足事实为假、与 gold/alias 不重合、答案类型/关系匹配和双语等价。
> 当前权威顺序见 [`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md)。

## 任务目标

优化distractor生成和验证流程，确保生成的干扰项具有合适的语义相似度，能够有效诱发模型翻转但不会过于简单或过于困难。

## 前置条件

- 可以与Task 1/2/3并行进行
- 如果Task 1发现distractor质量问题，此任务优先级提升

## 输入数据

**主要输入：**
- `data_processed/factual_triples/triple-full-v1/canonical-v2/distractor_candidates.jsonl` (7,976条现有distractors)
- Task 1或Task 2生成的distractors（如果已运行）

**参考数据：**
- v7b实验中的distractor效果数据
- 人工标注的distractor质量样本（如果有）

## 输出产物

1. **Distractor质量分析报告**
   - `tasks/reports/distractor_quality_analysis.md`
   - 现有distractors的语义相似度分布
   - 识别过于简单/困难的distractors

2. **增强的Distractor生成模块**
   - `scripts/generate_distractors_enhanced.py`
   - 集成语义相似度过滤
   - 支持answer_type特定的生成策略

3. **Distractor验证工具**
   - `scripts/validate_distractor_quality.py`
   - 自动化质量检查
   - 人工审核界面

## 详细步骤

### 1. 现有Distractor质量分析 (预计耗时: 2-3小时)

```bash
# 分析现有7,976条distractors
python3 scripts/analyze_distractor_quality.py \
  --input data_processed/factual_triples/triple-full-v1/canonical-v2/distractor_candidates.jsonl \
  --canonical_triples data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl \
  --output tasks/distractor_analysis/current_quality_report.json \
  --compute_embeddings \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

**分析维度：**
1. **语义相似度分布**
   - 与正确答案的余弦相似度
   - 期望范围：0.3-0.7
   - 识别过近（>0.7）和过远（<0.3）的distractors

2. **按answer_type分层**
   - 不同答案类型的distractor质量差异
   - 例如：person vs date vs concept

3. **Lexical overlap**
   - 与正确答案的字符串重叠
   - 避免仅改一个字的distractor

4. **Distractor多样性**
   - 同一base fact的多个distractors之间的相似度
   - 避免生成过于相似的干扰项

### 2. 问题诊断和根因分析 (预计耗时: 2小时)

```bash
# 提取问题案例
python3 scripts/extract_problematic_distractors.py \
  --quality_report tasks/distractor_analysis/current_quality_report.json \
  --output tasks/distractor_analysis/problematic_cases.jsonl \
  --threshold_too_similar 0.7 \
  --threshold_too_different 0.3
```

**问题分类：**
1. **过于简单**：与答案语义距离太远，模型不会被干扰
2. **过于困难**：与答案几乎相同，人类也难以区分
3. **语义无关**：完全不在同一语义空间
4. **类型不匹配**：例如用地名干扰人名问题

**根因假设：**
- 生成模型的temperature设置
- Prompt工程不足
- 缺少answer_type特定的约束
- 缺少后置过滤

### 3. 设计增强的生成策略 (预计耗时: 3小时)

**核心改进：**

#### 3.1 Answer_type特定的生成Prompt

```python
# 为不同答案类型设计专门的prompt
DISTRACTOR_PROMPTS = {
    "person": """生成3个与"{answer}"相似但不同的人名作为干扰项。
要求：
- 必须是真实存在的人
- 与正确答案在同一领域或职业
- 名气相近但不是正确答案
例如：正确答案是"爱因斯坦"，可以生成"牛顿"、"霍金"、"薛定谔"
""",
    
    "date": """生成3个与"{answer}"相近的日期作为干扰项。
要求：
- 时间差在合理范围内（±1-5年）
- 保持同样的历史时期
- 避免明显的历史标志性年份
""",
    
    "concept": """生成3个与"{answer}"语义相关但不同的概念作为干扰项。
要求：
- 在同一知识领域
- 语义相似度适中（不要太近也不要太远）
- 都是正确的概念，但不是这个问题的答案
"""
}
```

#### 3.2 语义相似度约束

```bash
# 使用embedding模型实时过滤
python3 scripts/generate_distractors_enhanced.py \
  --input data.jsonl \
  --output distractors.jsonl \
  --similarity_min 0.3 \
  --similarity_max 0.7 \
  --max_attempts_per_fact 10 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

#### 3.3 多样性约束

- 同一base fact的distractors之间相似度<0.6
- 避免生成重复的distractor

### 4. 实现增强的生成模块 (预计耗时: 4小时)

```bash
# 创建增强版distractor生成脚本
cat > scripts/generate_distractors_enhanced.py << 'PYEOF'
#!/usr/bin/env python3
"""
增强版distractor生成模块
特性：
1. Answer_type特定的prompt
2. 实时语义相似度过滤
3. 多样性约束
4. 质量自动评估
"""

import json
from sentence_transformers import SentenceTransformer
import numpy as np

class EnhancedDistractorGenerator:
    def __init__(self, similarity_min=0.3, similarity_max=0.7):
        self.similarity_min = similarity_min
        self.similarity_max = similarity_max
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate(self, fact, num_distractors=3):
        answer_type = fact['answer_type']
        prompt = self._get_prompt_for_type(answer_type, fact['answer'])
        
        candidates = []
        attempts = 0
        while len(candidates) < num_distractors and attempts < 20:
            # 调用LLM生成候选
            generated = self._call_llm(prompt)
            
            # 计算语义相似度
            similarity = self._compute_similarity(
                fact['answer'], 
                generated
            )
            
            # 过滤
            if self.similarity_min <= similarity <= self.similarity_max:
                # 检查与已有candidates的多样性
                if self._is_diverse(generated, candidates):
                    candidates.append({
                        'distractor': generated,
                        'similarity': similarity
                    })
            
            attempts += 1
        
        return candidates
PYEOF

# 测试增强模块
python3 scripts/generate_distractors_enhanced.py \
  --input tasks/distractor_analysis/problematic_cases.jsonl \
  --output tasks/distractor_analysis/regenerated_samples.jsonl \
  --num_samples 50
```

### 5. 质量验证和人工审核 (预计耗时: 3-4小时)

```bash
# 生成人工审核界面
python3 scripts/generate_distractor_review_interface.py \
  --original tasks/distractor_analysis/problematic_cases.jsonl \
  --regenerated tasks/distractor_analysis/regenerated_samples.jsonl \
  --output_html tasks/distractor_analysis/review_interface.html
```

**审核内容：**
- 对比原始distractor vs 增强生成的distractor
- 评估哪个更合适
- 记录审核结果

**审核样本：**
- 50条problematic cases的regenerated版本
- 随机抽取50条现有distractors作为baseline

**审核指标：**
1. **适当性** (1-5分)：distractor是否在合适的语义距离
2. **真实性** (1-5分)：distractor本身是否是真实/合理的
3. **多样性** (1-5分)：与其他distractors的区分度

### 6. A/B测试验证 (预计耗时: 需要实验数据)

**如果Task 1 pilot已完成：**
```bash
# 在pilot的子集上对比原始distractor vs 增强distractor
python3 scripts/run_distractor_ab_test.py \
  --pilot_results data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1 \
  --original_distractors data_processed/factual_triples/triple-full-v1/canonical-v2/distractor_candidates.jsonl \
  --enhanced_distractors tasks/distractor_analysis/regenerated_samples.jsonl \
  --output tasks/reports/distractor_ab_test.md
```

**对比指标：**
- PNT候选率（增强版 vs 原始版）
- 翻转强度（ZH targeted的准确率下降）
- Neutral稳定性

### 7. 生成最终报告和建议 (预计耗时: 2小时)

```bash
python3 scripts/generate_distractor_enhancement_report.py \
  --quality_analysis tasks/distractor_analysis/current_quality_report.json \
  --human_review tasks/distractor_analysis/review_results.jsonl \
  --ab_test tasks/reports/distractor_ab_test.md \
  --output tasks/reports/distractor_quality_enhancement.md
```

**报告内容：**
1. **现状评估**：现有distractors的质量分布
2. **问题诊断**：识别的主要问题和占比
3. **改进方案**：增强模块的设计和实现
4. **验证结果**：人工审核和A/B测试结果
5. **部署建议**：是否在Task 2中使用增强模块

## 质量门控

1. **语义相似度分布**：≥70%的distractors在0.3-0.7范围
2. **人工审核接受率**：增强版distractor接受率≥80%
3. **A/B测试提升**：如果可测，增强版PNT候选率提升≥10%
4. **多样性**：同一fact的distractors之间相似度<0.6

## 输出审核清单

**必须生成的文件：**
- [ ] `tasks/distractor_analysis/current_quality_report.json`
- [ ] `tasks/distractor_analysis/problematic_cases.jsonl`
- [ ] `scripts/generate_distractors_enhanced.py`
- [ ] `tasks/distractor_analysis/regenerated_samples.jsonl`
- [ ] `tasks/distractor_analysis/review_results.jsonl`
- [ ] `tasks/reports/distractor_quality_enhancement.md`

**报告必须回答：**
- 现有distractors的主要质量问题是什么？
- 增强模块相比原版有什么改进？
- 人工评估和A/B测试的结果如何？
- 是否建议在Task 2中使用增强模块？

## 依赖脚本

**需要创建的新脚本：**
- `scripts/analyze_distractor_quality.py`
- `scripts/extract_problematic_distractors.py`
- `scripts/generate_distractors_enhanced.py`
- `scripts/generate_distractor_review_interface.py`
- `scripts/run_distractor_ab_test.py` (可选)
- `scripts/generate_distractor_enhancement_report.py`

**依赖库：**
- `sentence-transformers` (embedding模型)
- `numpy`, `scipy` (相似度计算)

## 时间线

- **Day 1**: 质量分析 + 问题诊断 (步骤1-2)
- **Day 2**: 设计改进方案 + 实现增强模块 (步骤3-4)
- **Day 3**: 质量验证 + 人工审核 (步骤5)
- **Day 4**: A/B测试（如果可行）+ 报告生成 (步骤6-7)

**总计: 3-4个工作日**

## 成功标准

✅ 识别出现有distractors的质量问题
✅ 实现增强的生成模块，支持answer_type特定策略
✅ 语义相似度过滤有效（≥70%在合适范围）
✅ 人工审核验证增强版质量更高
✅ 提供明确的部署建议

## 后续衔接

**如果增强效果显著：**
- 在Task 2中使用增强模块生成distractors
- 考虑对现有7,976条distractors进行重新生成

**如果增强效果有限：**
- 保持现有pipeline，但增加人工审核环节
- 重点筛选answer≤2词且质量高的子集

## 可选增强

**如果时间和资源允许：**
1. **Domain-specific distractors**：为特定领域（科学、历史、文学）定制生成策略
2. **Adaptive difficulty**：根据模型能力动态调整distractor难度
3. **Cross-lingual consistency**：确保EN和ZH的distractors语义一致
4. **Distractor reuse**：在多个类似facts之间复用高质量distractors

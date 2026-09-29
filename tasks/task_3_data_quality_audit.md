# Task 3: 数据质量审计（REFERENCE ONLY）

> **状态：仅作参考，不是 formal admission。**  
> 抽样质量审计不能替代 160 条逐项 review、distractor review、leakage closure、
> `review_freeze_manifest` 或 `split_freeze_manifest`。
> 当前权威顺序见 [`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md)。

## 任务目标

对9,059条provisional数据和2,450条canonical数据进行人工抽检，评估数据质量，估计错误率，为后续数据使用决策提供依据。

## 前置条件

- 无前置依赖，可与Task 1/2并行进行
- 需要人工审核资源（约4-6小时审核时间）

## 输入数据

**主要输入：**
- `data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/provisional_triples.jsonl` (9,059条)
- `data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl` (2,450条)

## 输出产物

1. **审核数据集**
   - `tasks/quality_audit/provisional_sample_100.jsonl` (100条带审核结果)
   - `tasks/quality_audit/canonical_sample_100.jsonl` (100条带审核结果)

2. **质量评估报告**
   - `tasks/reports/data_quality_audit.md`
   - 包含：错误率估计、错误类型分布、质量分层建议

3. **决策建议**
   - 是否需要对9,059条补充dual-model验证
   - 是否可以直接使用2,450条进行扩展

## 详细步骤

### 1. 数据采样 (预计耗时: 15分钟)

```bash
# 创建审计目录
mkdir -p tasks/quality_audit

# 从9,059条provisional数据中分层采样50条
python3 scripts/sample_for_audit.py \
  --input data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/provisional_triples.jsonl \
  --output tasks/quality_audit/provisional_sample_50.jsonl \
  --sample_size 50 \
  --stratify_by source_dataset,answer_type \
  --random_seed 123

# 从2,450条canonical数据中分层采样50条
python3 scripts/sample_for_audit.py \
  --input data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl \
  --output tasks/quality_audit/canonical_sample_50.jsonl \
  --sample_size 50 \
  --stratify_by source_dataset,answer_type \
  --random_seed 456
```

**采样策略：**
- 按`source_dataset`分层（确保每个数据源都有覆盖）
- 按`answer_type`分层（确保不同答案类型都有覆盖）
- 优先采样answer≤2词的记录（这是后续扰动的重点）

### 2. 准备审核界面 (预计耗时: 30分钟)

```bash
# 生成人工审核的HTML界面
python3 scripts/generate_audit_interface.py \
  --provisional_sample tasks/quality_audit/provisional_sample_50.jsonl \
  --canonical_sample tasks/quality_audit/canonical_sample_50.jsonl \
  --output_html tasks/quality_audit/audit_interface.html
```

**审核界面功能：**
- 显示原始问题和选项
- 显示提取的subject, relation_raw, answer
- 提供判断选项：正确 / subject错误 / relation错误 / answer错误 / 多项错误
- 记录审核时间和审核者备注

### 3. 人工审核 (预计耗时: 4-6小时)

**审核标准：**

| 字段 | 判断标准 |
|---|---|
| **subject** | 是否准确提取了问题的主语/主体 |
| **relation_raw** | 关系描述是否准确反映了问题的语义 |
| **answer** | 是否与ground truth答案一致（考虑别名） |
| **整体质量** | 三元组是否构成一个有意义的事实陈述 |

**错误类型定义：**
1. **提取错误**：subject/relation/answer有一项或多项提取错误
2. **语义错误**：提取正确但三元组语义不完整或有歧义
3. **答案别名**：答案实质正确但表达不同（不算错误）
4. **问题缺陷**：原始问题本身有歧义或错误（标记但不算提取错误）

**审核流程：**
1. 打开`tasks/quality_audit/audit_interface.html`
2. 逐条审核100条数据（50条provisional + 50条canonical）
3. 对每条记录标注：正确 / 错误类型 / 备注
4. 保存审核结果到JSONL文件

### 4. 统计分析 (预计耗时: 30分钟)

```bash
# 合并审核结果
python3 scripts/analyze_audit_results.py \
  --provisional_audit tasks/quality_audit/provisional_sample_50_reviewed.jsonl \
  --canonical_audit tasks/quality_audit/canonical_sample_50_reviewed.jsonl \
  --output tasks/quality_audit/audit_statistics.json
```

**统计指标：**
- 整体错误率（provisional vs canonical）
- 错误类型分布（subject / relation / answer）
- 按数据集分层的错误率
- 按答案类型分层的错误率
- 短答案（≤2词）vs 长答案的错误率

### 5. 错误案例分析 (预计耗时: 1小时)

```bash
# 提取所有错误案例进行深入分析
python3 scripts/extract_error_cases.py \
  --provisional_audit tasks/quality_audit/provisional_sample_50_reviewed.jsonl \
  --canonical_audit tasks/quality_audit/canonical_sample_50_reviewed.jsonl \
  --output tasks/quality_audit/error_cases.jsonl
```

**分析维度：**
1. **系统性错误**：某个模型或某类问题的普遍问题
2. **随机错误**：偶发性错误
3. **可修复性**：是否可以通过规则或人工审核修正
4. **风险评估**：这些错误对下游扰动实验的影响

### 6. 生成审计报告 (预计耗时: 1小时)

```bash
# 生成完整的质量审计报告
python3 scripts/generate_quality_audit_report.py \
  --statistics tasks/quality_audit/audit_statistics.json \
  --error_cases tasks/quality_audit/error_cases.jsonl \
  --output tasks/reports/data_quality_audit.md
```

**报告内容：**

#### 6.1 执行摘要
- 采样方法和样本代表性
- 整体错误率估计（带置信区间）
- 关键发现和建议

#### 6.2 质量对比

| 数据集 | 样本数 | 错误数 | 错误率 | 95% CI |
|---|---:|---:|---:|---|
| Provisional (9,059条) | 50 | X | X% | [X%, Y%] |
| Canonical (2,450条) | 50 | Y | Y% | [X%, Y%] |

#### 6.3 错误类型分布

| 错误类型 | Provisional | Canonical |
|---|---:|---:|
| Subject错误 | X | Y |
| Relation错误 | X | Y |
| Answer错误 | X | Y |
| 多项错误 | X | Y |

#### 6.4 分层分析
- 按数据集（mkqa, global_mmlu等）
- 按答案类型（person, date, quantity等）
- 按答案长度（≤2词 vs >2词）

#### 6.5 错误案例
- 每个错误类型3-5个典型案例
- 根因分析
- 修复建议

#### 6.6 决策建议

**如果provisional错误率<10%：**
- 可以直接使用，无需额外验证
- 优先使用answer≤2词的子集

**如果provisional错误率10-20%：**
- 对高潜力候选（answer≤2词 + 高频关系）补充dual-model验证
- 或使用更严格的人工审核

**如果provisional错误率>20%：**
- 不建议直接使用
- 需要全量dual-model提取或放弃该数据源

**如果canonical错误率<5%：**
- 确认可以作为高质量扩展数据源
- 进入Task 5（2,450条预处理）

**如果canonical错误率≥5%：**
- 调查错误的根因（是提取问题还是原始数据问题）
- 决定是否需要重新审核

## 质量门控

1. **采样代表性**：卡方检验p>0.05（样本分布与总体一致）
2. **审核完整性**：100/100条完成审核
3. **审核一致性**：双人审核10条overlap，kappa≥0.75
4. **统计可信度**：置信区间宽度≤10%（对于错误率估计）

## 输出审核清单

**必须生成的文件：**
- [ ] `tasks/quality_audit/provisional_sample_50_reviewed.jsonl`
- [ ] `tasks/quality_audit/canonical_sample_50_reviewed.jsonl`
- [ ] `tasks/quality_audit/audit_statistics.json`
- [ ] `tasks/quality_audit/error_cases.jsonl`
- [ ] `tasks/reports/data_quality_audit.md`

**报告必须回答：**
- Provisional数据的错误率是多少？是否可用？
- Canonical数据的错误率是多少？是否可直接扩展？
- 主要的错误类型是什么？是否系统性？
- 是否需要补充验证？如果需要，具体方案是什么？

## 依赖脚本

**需要创建的新脚本：**
- `scripts/sample_for_audit.py`
- `scripts/generate_audit_interface.py`
- `scripts/analyze_audit_results.py`
- `scripts/extract_error_cases.py`
- `scripts/generate_quality_audit_report.py`

## 时间线

- **Day 1**: 采样 + 准备审核界面 (步骤1-2)
- **Day 2-3**: 人工审核 (步骤3)
- **Day 4**: 统计分析 + 错误分析 + 报告生成 (步骤4-6)

**总计: 3-4个工作日**

## 成功标准

✅ 100条数据完成审核
✅ 错误率估计有95%置信区间
✅ 识别出主要错误类型
✅ 提供明确的数据使用建议
✅ 如果需要补充验证，给出具体方案

## 后续衔接

**如果provisional质量可接受：**
- 可以进入Task 5（2,450条预处理），同时考虑从9,059条补充特定关系数据

**如果provisional质量不可接受：**
- 仅使用2,450条canonical数据
- 或对9,059条进行dual-model验证后再使用

**如果canonical质量可接受：**
- 确认进入Task 5（2,450条预处理和扩展）

## 审核资源需求

- **审核人员**：1-2人
- **审核时间**：每条约3-5分钟，100条共4-6小时
- **专业要求**：理解三元组提取任务，能判断subject/relation/answer的正确性
- **工具**：浏览器（审核界面）+ 文本编辑器（备注）

## 可选增强

**如果时间允许，可以额外审核：**
1. **双人审核overlap**：10-20条由两人独立审核，计算inter-rater kappa
2. **模型对比审核**：对比qwen3.7-plus vs sensenova-6.7-flash-lite在canonical数据上的差异
3. **关系质量审核**：对已映射到taxonomy的关系进行语义一致性检查

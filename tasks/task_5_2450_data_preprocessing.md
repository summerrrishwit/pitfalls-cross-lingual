# Task 5: 2,450条数据预处理和扩展（DEFERRED）

> **状态：阻塞，不属于当前 160 条闭环。**  
> 只有 160 条的 exact-HF attribution→repair、Validation、Sealed 和回归检查完成后，
> 才能另行设计并授权扩量。不得用扩量补救独立验证失败，也不得把 broad `mapped_relation`
> 当作全部扰动事实的准入条件。当前权威顺序见
> [`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md)。
> 本文以下内容仅保留为历史设计记录，不得按其中命令执行。

## 任务目标

将2,450条canonical数据预处理为扰动实验输入格式，作为Task 2的扩展，目标产出100-200条高质量PNT候选数据。

## 前置条件

- **Task 2必须成功完成**：160条已产出30-50条PNT候选，验证了扰动策略有效性
- **Task 3的质量审计通过**：canonical数据错误率<5%

## 输入数据

**主要输入：**
- `data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl` (2,450条)
- `data_processed/factual_triples/triple-full-v1/canonical-v2/factual_prompts.jsonl` (2,450条)

**参考配置：**
- Task 2的成功配置和参数
- Task 4的distractor增强模块（如果可用）

## 输出产物

1. **筛选后的高质量候选池**
   - `data_processed/factual_perturbation/canonical-expansion-v1/filtered_candidates.jsonl` (~1,000条)
   - 筛选标准：answer≤2词 + mapped relation

2. **完整扰动实验结果**
   - `data_processed/factual_perturbation/canonical-expansion-v1/runs/full-expansion-v1/`
   - 实验报告、三臂分析

3. **PNT候选数据（累积）**
   - `data_processed/path_not_token/chinese-v1-enhanced/base_facts_extended.jsonl` (130-250条)
   - 包含Task 2的30-50条 + 本任务的100-200条

## 详细步骤

### 1. 数据筛选和优先级排序 (预计耗时: 2小时)

```bash
# 从2,450条中筛选高质量候选
python3 scripts/filter_canonical_for_perturbation.py \
  --input data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl \
  --factual_prompts data_processed/factual_triples/triple-full-v1/canonical-v2/factual_prompts.jsonl \
  --relation_mapping data_processed/factual_triples/triple-full-v1/canonical-v2/relation_mapping_v1.json \
  --output data_processed/factual_perturbation/canonical-expansion-v1/filtered_candidates.jsonl \
  --criteria answer_length_max_2,mapped_relation,strict_factual_completion \
  --priority_by answer_length,relation_frequency
```

**筛选标准：**
1. **Answer长度**：≤2词（当前2,109条，86.1%符合）
2. **Relation映射**：有mapped relation（约60%）
3. **Prompt质量**：`prompt_quality_tier = "strict_factual_completion"` (1,590条)

**交集估算：**
- 2,109条（≤2词）× 0.6（mapped）× 0.65（strict_factual）≈ 820-900条

**优先级排序：**
- 优先answer=1词（最适合开放补全）
- 优先高频关系（与PNT原始关系对齐）
- 优先sciq/ai2_arc_easy数据集（科学事实类）

### 2. 分批处理策略 (预计总耗时: 3-4周)

考虑到规模和成本，采用分批渐进策略：

#### Batch 1: 200条初始批次 (Week 1)
```bash
python3 scripts/sample_batch.py \
  --input data_processed/factual_perturbation/canonical-expansion-v1/filtered_candidates.jsonl \
  --output data_processed/factual_perturbation/canonical-expansion-v1/batch_1_200.jsonl \
  --size 200 \
  --ensure_relation_diversity
```

**目标：**
- 验证在更大规模上扩展的可行性
- 估计PNT候选率是否保持在Task 2的20%水平
- 预期产出：40条PNT候选

#### Batch 2: 300条扩展批次 (Week 2-3，依赖Batch 1成功)
```bash
python3 scripts/sample_batch.py \
  --input data_processed/factual_perturbation/canonical-expansion-v1/filtered_candidates.jsonl \
  --output data_processed/factual_perturbation/canonical-expansion-v1/batch_2_300.jsonl \
  --size 300 \
  --exclude_batch_1 \
  --prioritize_underrepresented_relations
```

**目标：**
- 补充Batch 1中覆盖不足的关系
- 预期产出：60条PNT候选

#### Batch 3: 剩余候选（可选，Week 4+）
- 仅在前两批成功且需要更多候选时执行
- 或在特定关系数据不足时补充

### 3. Translation和Distractor生成 (每批: 1-2天)

**复用Task 2的优化流程：**

```bash
# Translation (使用现有answer_zh，仅需验证和补充)
python3 scripts/prepare_canonical_translations.py \
  --input data_processed/factual_perturbation/canonical-expansion-v1/batch_1_200.jsonl \
  --canonical_triples data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl \
  --output data_processed/factual_perturbation/canonical-expansion-v1/batch_1_200_with_translations.jsonl \
  --reuse_existing_translations

# Distractor生成（使用Task 4的增强模块，如果可用）
python3 scripts/generate_distractors_enhanced.py \
  --input data_processed/factual_perturbation/canonical-expansion-v1/batch_1_200_with_translations.jsonl \
  --output data_processed/factual_perturbation/canonical-expansion-v1/batch_1_200_complete.jsonl \
  --similarity_min 0.3 \
  --similarity_max 0.7 \
  --num_distractors 4
```

**优势：**
- Canonical数据已有answer_zh（qwen3.8-max + gpt-5.5复核）
- 只需验证质量，无需重新翻译
- 节省时间和成本

### 4. 三臂扰动实验 (每批: 2-3天)

**复用Task 2的模型配置和参数：**

```bash
# Batch 1实验
python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_canonical_batch_1.json \
  --input data_processed/factual_perturbation/canonical-expansion-v1/batch_1_200_complete.jsonl \
  --output_dir data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-1-200-v1 \
  --enable_neutral_control \
  --neutral_stability_threshold 2 \
  --resume_from_checkpoint
```

**实验规模（每批200条）：**
- 200条 × 6条件 × 5模型 = 6,000个测量点
- API成本：~$150-200/批次
- 耗时：16-20小时/批次

### 5. 结果整合和PNT候选合并 (预计耗时: 1天)

```bash
# 合并Task 2和Task 5的PNT候选
python3 scripts/merge_pnt_candidates.py \
  --task2_candidates data_processed/path_not_token/chinese-v1-enhanced/base_facts.jsonl \
  --batch1_candidates data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-1-200-v1/pnt_candidates.jsonl \
  --batch2_candidates data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-2-300-v1/pnt_candidates.jsonl \
  --output data_processed/path_not_token/chinese-v1-enhanced/base_facts_extended.jsonl \
  --deduplicate_by source_id
```

**整合检查：**
- 去重（基于source_id）
- 确保关系分布合理
- 验证schema一致性

### 6. 最终评估报告 (预计耗时: 1天)

```bash
python3 scripts/generate_expansion_assessment.py \
  --task2_results data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1 \
  --batch1_results data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-1-200-v1 \
  --batch2_results data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-2-300-v1 \
  --merged_pnt data_processed/path_not_token/chinese-v1-enhanced/base_facts_extended.jsonl \
  --output tasks/reports/canonical_expansion_assessment.md
```

**报告内容：**
1. 筛选统计（从2,450到~1,000）
2. 分批执行概览
3. PNT候选累积统计（160条 → 200条 → 500条）
4. 关系覆盖分析
5. 数据集来源分析
6. 成本和时间总结
7. 与9,059条provisional数据的对比
8. 后续建议（是否需要补充9,059条数据）

## 质量门控

**Batch 1（200条）：**
1. 完整模拟率≥90%
2. PNT候选率≥15%（允许比Task 2略低）
3. Neutral不稳定率≤15%
4. Translation质量：复用率≥95%

**整体（累积）：**
1. 总PNT候选数≥100条
2. 关系覆盖：至少覆盖10个不同关系
3. 数据集来源多样性：至少覆盖3个数据集
4. 成本控制：≤$600（所有批次）

## 决策点和Go/No-go

**在Batch 1完成后：**
- 如果PNT候选率<10%，停止扩展，调查原因
- 如果成本超预算，调整Batch 2规模或暂停
- 如果Neutral不稳定率>20%，先解决稳定性问题（Task 6）

**在Batch 2完成后：**
- 评估是否已达到100条PNT候选目标
- 如果已达标，停止扩展
- 如果未达标但接近（80-90条），评估是否补充Batch 3

## 输出审核清单

**必须生成的文件（每批）：**
- [ ] `data_processed/factual_perturbation/canonical-expansion-v1/batch_X_XXX_complete.jsonl`
- [ ] `data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-X-XXX-v1/experiment_report.md`
- [ ] `data_processed/factual_perturbation/canonical-expansion-v1/runs/batch-X-XXX-v1/pnt_candidates.jsonl`

**最终整合文件：**
- [ ] `data_processed/path_not_token/chinese-v1-enhanced/base_facts_extended.jsonl`
- [ ] `tasks/reports/canonical_expansion_assessment.md`

**报告必须回答：**
- 从2,450条筛选出多少候选？
- 每批的PNT候选率如何？
- 累积PNT候选数是否达到100条目标？
- 关系和数据集分布如何？
- 是否需要补充9,059条provisional数据？

## 依赖脚本

**需要创建的新脚本：**
- `scripts/filter_canonical_for_perturbation.py`
- `scripts/sample_batch.py`
- `scripts/prepare_canonical_translations.py`
- `scripts/merge_pnt_candidates.py`
- `scripts/generate_expansion_assessment.py`

**复用的现有脚本：**
- `scripts/generate_distractors_enhanced.py` (Task 4)
- `scripts/run_factual_perturbation.py`
- `scripts/analyze_three_arm_results.py`

## 时间线

**保守估计（执行Batch 1 + Batch 2）：**
- **Week 1**: 筛选 + Batch 1 translation + distractor + 扰动实验
- **Week 2**: Batch 1分析 + go/no-go决策 + Batch 2准备
- **Week 3**: Batch 2 translation + distractor + 扰动实验
- **Week 4**: Batch 2分析 + 整合 + 最终报告

**总计: 3-4周**

**如果只执行Batch 1：**
- **总计: 1周**

## 成功标准

✅ 从2,450条筛选出≥800条高质量候选
✅ Batch 1产出≥30条PNT候选
✅ 累积PNT候选数≥100条（包含Task 2）
✅ 关系覆盖≥10个
✅ 成本控制在$600以内
✅ 数据已整合为统一PNT格式

## 后续衔接

**如果成功达到100条PNT候选：**
- 停止扩展，进入PNT baseline实验（外部任务）
- 9,059条provisional数据作为未来补充备选

**如果未达到100条但接近：**
- 评估是否补充Batch 3（canonical剩余候选）
- 或从9,059条中补充特定关系的高质量数据

**如果PNT候选率显著下降：**
- 调查canonical数据与160条relation-balanced数据的差异
- 可能需要Task 6（Neutral稳定性优化）

## 风险缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| Batch 1 PNT候选率低于预期 | 中 | 高 | 调查根因，调整筛选标准或停止扩展 |
| 成本超预算 | 中 | 中 | 分批控制，每批后评估ROI |
| Neutral不稳定率上升 | 低 | 中 | 已在Task 2验证，使用相同配置 |
| 数据筛选过于严格 | 低 | 中 | Batch 1后可放宽标准 |

## 可选优化

**如果时间和资源允许：**
1. **并行处理**：Batch 1和Batch 2的translation可以并行
2. **Adaptive sampling**：根据Batch 1结果动态调整Batch 2的采样策略
3. **从9,059条补充**：如果canonical数据不足，补充high-confidence provisional数据
4. **Cross-validation**：在不同数据集上验证扰动策略的泛化性

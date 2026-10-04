# Task 1: 160条数据扰动pilot验证（SUPERSEDED）

> **历史记录，非执行规范。正文故意保留旧设计，不代表当前状态或已完成工作，任何命令、预算、阈值、成功标记均不得执行或引用。**  
> 当前唯一权威设计见 [`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md)：双 distractor、活动四代理仅运行 Development 96、3,840 个主 logical calls、逐模型两级 proxy 标签、Codex proxy 非人工金标；`G0A` 已冻结，活动配置为 `configs/factual_perturbation_zh_dual_distractor_proxy_v10.json`，run 为 `static-g0a-dev96-four-model-v10-20260916`，Gemini `max_output_tokens=512`。P0 已完成并审计为 3,840/3,840、`proxy_screen_complete=true`；`gate_passed=false` 仅因 `proxy_behavior_only_not_pnt_eligible`，不解锁 PNT。
> 原因：本文会从全部 160 条随机抽取 50 条、直接使用 review-only 输入，并把 MCQ flip 当作 PNT 候选，
> 会消费 Validation/Sealed、混淆模型身份并绕过 formal review/freeze。
> 本文以下内容仅保留为历史设计记录。

## 任务目标

在160条relation-balanced数据上完成小规模扰动pilot，验证"扰动→badcase→PNT修复"全流程的可行性。

## 前置条件

- 已有160条relation-balanced数据：`data_processed/path_not_token/public-benchmark-qwen-postclosure-provisional-v2/review_only_base_facts.jsonl`
- 当前状态：pending_review，未完成translation/split freeze
- 4个已选定关系的候选数据

## 输入数据

**主要输入：**
- `data_processed/path_not_token/public-benchmark-qwen-postclosure-provisional-v2/review_only_base_facts.jsonl` (160条)

**参考配置：**
- `configs/factual_perturbation_zh_five_model_v7.json`
- `data_processed/factual_perturbation/zh-mvp-v1/runs/diagnostic-100-five-model-v7b/` (v7b实验结果)

## 输出产物

1. **Pilot测试结果**
   - `data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1/`
   - 包含：实验报告、三臂分析、PNT候选清单

2. **质量评估报告**
   - `tasks/reports/pilot_50_assessment.md`
   - 统计：PNT候选数、翻转率、Neutral不稳定性

3. **Go/No-go决策**
   - 如果PNT候选数 ≥10条 → 进入Task 2（全量160条）
   - 如果候选数 <10条 → 需要调整扰动策略或数据筛选

## 详细步骤

### 1. 数据采样 (预计耗时: 10分钟)

```bash
# 从160条中随机抽取50条
python3 scripts/sample_pilot_data.py \
  --input data_processed/path_not_token/public-benchmark-qwen-postclosure-provisional-v2/review_only_base_facts.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/base_facts_sample.jsonl \
  --sample_size 50 \
  --random_seed 42 \
  --ensure_relation_coverage
```

**验证点：**
- 确保4个关系都有覆盖（每个关系至少10条）
- 输出采样统计报告

### 2. Translation Review (预计耗时: 2-3小时)

**目标：** 补充50条数据的中文翻译和验证

```bash
# 使用qwen3.8-max进行翻译
python3 scripts/run_translation_review.py \
  --input data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/base_facts_sample.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/translations_v1.jsonl \
  --model qwen3.8-max \
  --target_language zh \
  --batch_size 10

# 使用gpt-5.5进行复核
python3 scripts/run_translation_verification.py \
  --translations data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/translations_v1.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/translations_verified.jsonl \
  --model gpt-5.5 \
  --verification_criteria semantic_equivalence
```

**质量门槛：**
- 翻译一致性 >90%
- 语义等价性人工抽检（10条）

**成本估算：**
- qwen3.8-max: ~50条 × 200 tokens = 10K tokens (~$0.5)
- gpt-5.5: ~50条 × 150 tokens = 7.5K tokens (~$2)
- 总计: ~$2.5

### 3. Distractor生成 (预计耗时: 1小时)

```bash
# 复用现有distractor生成pipeline
python3 scripts/generate_distractors.py \
  --input data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/translations_verified.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/triples_with_distractors.jsonl \
  --num_distractors 3 \
  --generation_model qwen3.7-plus \
  --similarity_threshold_min 0.3 \
  --similarity_threshold_max 0.7
```

**验证点：**
- 每条至少3个distractors
- 语义相似度在0.3-0.7之间
- 人工抽检10条确保质量

### 4. 三臂扰动实验 (预计耗时: 4-6小时)

**模型配置（基于v7b优化）：**
- 使用HuggingFace版本替代Ollama
- 统一设置：`temperature=0, top_p=1`
- 5个测试模型：
  - `qwen3.6-27b` (Ollama)
  - `bailian/deepseek-v3.2` (API)
  - `gemini-3.7-flash` (API)
  - `gemma-2-9b` (HF, 替代Ollama gemma3:12b)
  - `llama-3.1-8b` (HF, 替代Ollama版本)

```bash
# 运行三臂实验
python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_pilot_50_v1.json \
  --input data_processed/factual_perturbation/zh-mvp-v1/pilot-50-v1/triples_with_distractors.jsonl \
  --output_dir data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1 \
  --enable_neutral_control \
  --neutral_stability_threshold 2
```

**三臂设计：**
- EN_original / EN_neutral / EN_targeted
- ZH_original / ZH_neutral / ZH_targeted
- 每个base fact × 6个条件 × 5个模型 = 30个测量点

**成本估算：**
- API调用: 50条 × 30点 × (平均500 tokens) = 750K tokens (~$30-50)
- Ollama: 本地免费
- 总计: ~$30-50

### 5. PNT候选筛选 (预计耗时: 30分钟)

**筛选标准（严格版）：**
1. **Strict targeted flip:** EN_targeted正确 且 ZH_targeted错误
2. **Neutral stability:** Neutral条件下，5个模型中≤1个出现不稳定
3. **Distractor特异性:** 翻转必须指向planted distractor
4. **模型特异性:** 至少在1个模型上出现信号

```bash
# 运行三臂分析和PNT候选筛选
python3 scripts/analyze_three_arm_results.py \
  --input_dir data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1 \
  --output data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1/three_arm_analysis.json \
  --strict_mode \
  --neutral_instability_threshold 1
```

**输出统计：**
- 完整模拟的候选数
- PNT候选数和百分比
- 每个模型的strict signal count
- Neutral不稳定案例数

### 6. 结果评估和决策 (预计耗时: 1小时)

生成评估报告：

```bash
python3 scripts/generate_pilot_assessment.py \
  --three_arm_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1/three_arm_analysis.json \
  --output tasks/reports/pilot_50_assessment.md
```

**决策标准：**

| 指标 | 目标 | Go/No-go |
|---|---|---|
| PNT候选数 | ≥10条 | Go to Task 2 |
| PNT候选率 | ≥20% | Go to Task 2 |
| Neutral不稳定率 | ≤10% | Go to Task 2 |
| 翻转率（ZH targeted） | 平均下降≥10% | Go to Task 2 |

**如果No-go，诊断选项：**
- 扰动强度不足 → 增加distractor相似度
- 数据质量问题 → 补充dual-model验证
- 模型选择问题 → 调整模型组合
- 关系不适配 → 重新筛选关系

## 质量门控

1. **Translation完整性**: 50/50条通过验证
2. **Distractor质量**: 人工抽检10条，接受率≥80%
3. **实验完整性**: 完整模拟率≥90% (45/50)
4. **成本门控**: 总成本≤$60

## 输出审核

**生成以下文档：**
1. `tasks/reports/pilot_50_assessment.md` - 完整评估报告
2. `data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1/experiment_report.md`
3. `data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1/pnt_candidates.jsonl`

**报告必须包含：**
- Go/No-go明确决策
- 如果Go：估计全量160条的PNT候选数
- 如果No-go：根因分析和优化建议

## 依赖脚本

**需要创建的新脚本：**
- `scripts/sample_pilot_data.py`
- `scripts/generate_pilot_assessment.py`

**复用的现有脚本：**
- `scripts/run_translation_review.py`
- `scripts/generate_distractors.py`
- `scripts/run_factual_perturbation.py`
- `scripts/analyze_three_arm_results.py`

## 时间线

- Day 1: 数据采样 + Translation (步骤1-2)
- Day 2: Distractor生成 + 开始扰动实验 (步骤3-4)
- Day 3: 完成扰动实验 + 分析 (步骤4-5)
- Day 4: 结果评估和报告 (步骤6)

**总计: 3-4个工作日**

## 成功标准

✅ 获得≥10条严格的PNT候选
✅ Neutral不稳定率≤10%
✅ 在4个关系上都有信号
✅ 成本控制在$60以内
✅ 生成完整的go/no-go决策报告

## 后续衔接

- **如果成功**: 进入Task 2（160条全量扰动）
- **如果失败**: 迭代优化后重新pilot，或调整整体策略

# Task 2: 160条数据全量扰动实验（SUPERSEDED）

> **历史记录，非执行规范。正文故意保留旧设计，不代表当前状态或已完成工作，任何命令、预算、阈值、成功标记均不得执行或引用。**  
> 当前唯一权威设计见 [`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md)：双 distractor、活动四代理仅运行 Development 96、3,840 个主 logical calls、逐模型两级 proxy 标签、Codex proxy 非人工金标；`G0A` 已冻结，活动配置为 `configs/factual_perturbation_zh_dual_distractor_proxy_v10.json`，run 为 `static-g0a-dev96-four-model-v10-20260916`，Gemini `max_output_tokens=512`。P0 已完成并审计为 3,840/3,840、`proxy_screen_complete=true`；`gate_passed=false` 仅因 `proxy_behavior_only_not_pnt_eligible`，不解锁 PNT。
> 原因：本文混合使用 Development/Validation/Sealed，把多模型 MCQ 结果直接导出为 PNT 数据，
> 并引用尚不存在或 CLI 不兼容的脚本。
> 本文以下内容仅保留为历史设计记录。

## 任务目标

在Task 1 pilot成功的基础上，完成160条relation-balanced数据的全量扰动实验，产出30-50条高质量PNT候选数据。

## 前置条件

- **Task 1必须成功**：pilot获得≥10条PNT候选，Neutral不稳定率≤10%
- Task 1的go决策报告已生成
- 扰动策略和参数已经过pilot验证

## 输入数据

**主要输入：**
- `data_processed/path_not_token/public-benchmark-qwen-postclosure-provisional-v2/review_only_base_facts.jsonl` (160条)

**参考配置：**
- `tasks/reports/pilot_50_assessment.md` (Task 1的决策依据)
- `data_processed/factual_perturbation/zh-mvp-v1/runs/pilot-50-relation-balanced-v1/` (pilot实验配置)

## 输出产物

1. **完整扰动实验结果**
   - `data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/`
   - 包含：实验报告、三臂分析、完整的PNT候选数据

2. **PNT输入数据**
   - `data_processed/path_not_token/chinese-v1-enhanced/base_facts.jsonl` (30-50条)
   - 符合PNT输入schema的标准格式

3. **质量评估报告**
   - `tasks/reports/full_160_assessment.md`
   - 关系分层分析、模型效果对比

## 详细步骤

### 1. 全量Translation Review (预计耗时: 6-8小时)

```bash
# 使用qwen3.8-max翻译全部160条
python3 scripts/run_translation_review.py \
  --input data_processed/path_not_token/public-benchmark-qwen-postclosure-provisional-v2/review_only_base_facts.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/translations_v1.jsonl \
  --model qwen3.8-max \
  --target_language zh \
  --batch_size 20 \
  --resume_from_checkpoint

# 使用gpt-5.5复核
python3 scripts/run_translation_verification.py \
  --translations data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/translations_v1.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/translations_verified.jsonl \
  --model gpt-5.5 \
  --verification_criteria semantic_equivalence \
  --flag_low_confidence
```

**质量门槛：**
- 翻译完整率: 160/160
- 语义等价率: ≥95%
- 低置信度标记: 人工复核

**成本估算：**
- qwen3.8-max: 160条 × 200 tokens = 32K tokens (~$1.5)
- gpt-5.5: 160条 × 150 tokens = 24K tokens (~$6)
- 总计: ~$7.5

### 2. Distractor生成和质量验证 (预计耗时: 3-4小时)

```bash
# 生成distractors（应用Task 1学到的相似度阈值）
python3 scripts/generate_distractors.py \
  --input data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/translations_verified.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/distractors_raw.jsonl \
  --num_distractors 4 \
  --generation_model qwen3.7-plus \
  --similarity_threshold_min 0.3 \
  --similarity_threshold_max 0.7 \
  --use_semantic_filtering

# 质量验证和人工抽检
python3 scripts/validate_distractors.py \
  --input data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/distractors_raw.jsonl \
  --output data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/triples_with_distractors.jsonl \
  --min_distractors_per_fact 3 \
  --sample_for_review 20
```

**验证点：**
- 每条base fact至少3个有效distractors
- 语义相似度分布符合预期（0.3-0.7）
- 人工抽检20条（覆盖4个关系）

### 3. 三臂扰动实验 (预计耗时: 12-16小时)

**模型配置（与pilot一致）：**
- HuggingFace版本优先
- `temperature=0, top_p=1`
- 5个模型：qwen3.6-27b (Ollama), bailian/deepseek-v3.2, gemini-3.7-flash, gemma-2-9b (HF), llama-3.1-8b (HF)

```bash
# 创建实验配置
cat > configs/factual_perturbation_full_160_v1.json << 'INNER_EOF'
{
  "experiment_name": "full-160-relation-balanced-v1",
  "base_config": "factual_perturbation_pilot_50_v1.json",
  "input_size": 160,
  "models": [
    {"name": "qwen3.6-27b", "provider": "ollama", "temperature": 0},
    {"name": "bailian/deepseek-v3.2", "provider": "api", "temperature": 0},
    {"name": "gemini-3.7-flash", "provider": "api", "temperature": 0},
    {"name": "gemma-2-9b", "provider": "hf", "temperature": 0},
    {"name": "llama-3.1-8b", "provider": "hf", "temperature": 0}
  ],
  "arms": ["en_original", "en_neutral", "en_targeted", "zh_original", "zh_neutral", "zh_targeted"],
  "neutral_stability_threshold": 2,
  "enable_checkpointing": true,
  "checkpoint_interval": 20
}
INNER_EOF

# 运行全量实验
python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_full_160_v1.json \
  --input data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/triples_with_distractors.jsonl \
  --output_dir data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1 \
  --enable_neutral_control \
  --resume_from_checkpoint
```

**实验规模：**
- 160条 × 6条件 × 5模型 = 4,800个测量点
- 预计耗时：12-16小时（取决于API响应速度）

**成本估算：**
- API调用: 4,800点 × 500 tokens = 2.4M tokens (~$100-150)
- HF模型: 本地GPU，需要约16GB显存
- Ollama: 本地免费

**监控点：**
- 每20条输出checkpoint
- 实时监控Neutral不稳定性
- API失败自动重试（最多3次）

### 4. 三臂分析和PNT候选筛选 (预计耗时: 1-2小时)

```bash
# 运行三臂分析
python3 scripts/analyze_three_arm_results.py \
  --input_dir data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1 \
  --output data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/three_arm_analysis.json \
  --strict_mode \
  --neutral_instability_threshold 1 \
  --enable_relation_stratification

# 生成关系分层报告
python3 scripts/generate_relation_stratified_report.py \
  --three_arm_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/three_arm_analysis.json \
  --output data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/relation_analysis.json
```

**筛选标准（严格版，与pilot一致）：**
1. Strict targeted flip: EN_targeted正确 且 ZH_targeted错误
2. Neutral stability: ≤1个模型出现不稳定
3. Distractor特异性
4. 模型特异性

**预期输出：**
- 基于pilot的20%候选率，预期32条PNT候选
- 目标范围：30-50条
- 每个关系至少6条

### 5. 导出PNT格式数据 (预计耗时: 1小时)

```bash
# 创建PNT schema导出脚本
python3 scripts/export_pnt_candidates.py \
  --three_arm_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/three_arm_analysis.json \
  --input_bundle data_processed/factual_perturbation/zh-mvp-v1/full-160-v1/triples_with_distractors.jsonl \
  --output_dir data_processed/path_not_token/chinese-v1-enhanced \
  --schema_version path-not-token-input-v1
```

**输出schema字段：**
- `known_id`: 基于source_id的稳定哈希
- `subject`, `relation_id`, `prompt`, `answer`
- `answer_aliases`: EN + ZH别名
- `translation`: answer_zh (已验证)
- `context_en`, `context_zh`: 扰动上下文（如果有）
- `pitfall_score`: 基于翻转强度
- `badcase_type`: "induced" (扰动产生)
- `source_metadata`: 溯源信息

### 6. 质量评估和报告生成 (预计耗时: 2小时)

```bash
# 生成完整评估报告
python3 scripts/generate_full_assessment.py \
  --three_arm_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/three_arm_analysis.json \
  --relation_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/relation_analysis.json \
  --pnt_candidates data_processed/path_not_token/chinese-v1-enhanced/base_facts.jsonl \
  --output tasks/reports/full_160_assessment.md
```

**报告内容：**
1. 实验概览（完整率、成本、耗时）
2. PNT候选统计（总数、关系分布、模型分布）
3. 关系分层分析（每个关系的效果）
4. 与v7b对比（候选率、不稳定性）
5. 质量门控检查结果
6. 后续建议

## 质量门控

1. **实验完整性**: ≥150/160完整模拟 (93.75%)
2. **Translation质量**: 人工抽检20条，接受率≥90%
3. **Distractor质量**: 人工抽检20条，接受率≥85%
4. **Neutral稳定性**: 不稳定率≤15% (对比pilot的目标≤10%)
5. **PNT候选数**: 30-50条
6. **关系覆盖**: 4个关系都有候选
7. **成本门控**: 总成本≤$200

## 风险缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| API配额耗尽 | 中 | 高 | 分批运行，设置daily limit |
| Neutral不稳定率超标 | 低 | 中 | 已在pilot中验证，使用HF版本 |
| PNT候选数不足 | 低 | 高 | Pilot已验证20%率，160条足够 |
| 成本超预算 | 中 | 中 | 实时监控，可暂停调整 |
| GPU显存不足 | 低 | 中 | HF模型使用bf16精度 |

## 输出审核清单

**必须生成的文件：**
- [ ] `data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/experiment_report.md`
- [ ] `data_processed/factual_perturbation/zh-mvp-v1/runs/full-160-relation-balanced-v1/three_arm_analysis.json`
- [ ] `data_processed/path_not_token/chinese-v1-enhanced/base_facts.jsonl`
- [ ] `tasks/reports/full_160_assessment.md`

**报告必须回答：**
- 是否达到30-50条PNT候选？
- 每个关系的效果如何？
- 与v7b的14条相比，扰动策略是否改进？
- 是否ready进入PNT baseline实验（Task 5/6）？

## 依赖脚本

**需要创建的新脚本：**
- `scripts/validate_distractors.py`
- `scripts/generate_relation_stratified_report.py`
- `scripts/export_pnt_candidates.py`
- `scripts/generate_full_assessment.py`

**复用的现有脚本：**
- `scripts/run_translation_review.py`
- `scripts/generate_distractors.py`
- `scripts/run_factual_perturbation.py`
- `scripts/analyze_three_arm_results.py`

## 时间线

- **Week 1, Day 1-2**: Translation review全量完成
- **Week 1, Day 3**: Distractor生成和验证
- **Week 1, Day 4 - Week 2, Day 1**: 三臂扰动实验（可能需要分批）
- **Week 2, Day 2**: 分析、导出PNT格式
- **Week 2, Day 3**: 质量评估和报告

**总计: 8-10个工作日**

## 成功标准

✅ 获得30-50条严格的PNT候选
✅ 4个关系全部覆盖，每个关系≥6条
✅ Neutral不稳定率≤15%
✅ 完整模拟率≥93%
✅ 成本控制在$200以内
✅ 数据已导出为PNT标准格式
✅ 质量评估报告完整

## 后续衔接

**Task 3**: 数据质量审计（可并行）
**Task 4**: Distractor质量增强（如果本任务发现质量问题）
**Task 5**: PNT baseline实验（需要本任务的30-50条候选）
**Task 6**: Neutral稳定性优化（如果不稳定率超标）

## 关键决策点

**在步骤3之前：**
- 确认GPU资源可用（HF模型需要）
- 确认API配额充足
- 确认成本预算批准

**在步骤5之前：**
- 如果PNT候选数<25条，需要调查原因并决定是否调整筛选标准
- 如果某个关系候选数<3条，考虑补充该关系的数据

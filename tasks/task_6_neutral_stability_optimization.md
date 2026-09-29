# Task 6: Neutral稳定性优化（HISTORICAL REFERENCE）

> **状态：旧 73 条诊断的校准草案，正文命令、阈值和“优化”决策不得直接用于当前 160。**  
> 当前协议已经授权的“strict 放宽”仅指把主挖掘 gate 改为单模型、单 distractor 的
> `directed_candidate` 中文三臂条件；英文三臂只决定更窄的 `zh_specific_strict`。
> 不得进一步删除 Neutral、放宽 designated-distractor hit、按结果改 context，或切换模型身份来保住候选数量；
> 任何静态内容变化都必须发布新的 G0A 版本并完整重跑 P0，更换 checkpoint 会产生新的模型实验链。当前权威顺序见
> [`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md)。

## 任务目标

消除v7b实验中发现的41个Neutral-control不稳定案例，确保三臂扰动实验的Neutral条件能够提供可靠的基线，为PNT候选筛选提供稳定的判断依据。

## 前置条件

- v7b实验已完成，识别出41个Neutral不稳定案例
- 如果Task 1或Task 2发现Neutral不稳定率>15%，此任务优先级提升

## 输入数据

**主要输入：**
- `data_processed/factual_perturbation/zh-mvp-v1/runs/diagnostic-100-five-model-v7b/three_arm_analysis.json`
- v7b的41个不稳定案例详细信息

**对照数据：**
- v5c实验（Neutral不稳定率为0）的配置和结果

## 输出产物

1. **不稳定性分析报告**
   - `tasks/reports/neutral_instability_analysis.md`
   - 41个案例的模式分析
   - 根因假设

2. **优化后的模型配置**
   - `configs/factual_perturbation_stable_neutral_v1.json`
   - 经过验证的稳定配置

3. **验证实验结果**
   - `data_processed/factual_perturbation/zh-mvp-v1/runs/neutral-stability-verification-v1/`
   - 在v7b的73条上重跑验证

## 详细步骤

### 1. 不稳定案例分析 (预计耗时: 2-3小时)

```bash
# 提取所有Neutral不稳定案例
python3 scripts/extract_neutral_instability_cases.py \
  --three_arm_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/diagnostic-100-five-model-v7b/three_arm_analysis.json \
  --output tasks/neutral_stability/instability_cases.jsonl \
  --include_full_context
```

**分析维度：**

1. **按模型分布**
   - 哪些模型更容易出现不稳定？
   - Ollama版本 vs HF版本的差异

2. **按base fact特征**
   - Answer类型：person, date, concept等
   - Answer长度
   - Prompt类型
   - 数据集来源

3. **按条件分布**
   - EN_neutral vs ZH_neutral
   - Original vs Neutral的差异模式

4. **时序分析**
   - 不稳定是否集中在某个时间段（API服务波动？）
   - 是否与retry相关

### 2. 根因假设和验证 (预计耗时: 3-4小时)

**假设1: Ollama量化导致的不确定性**

v7b使用Ollama版本：
- `gemma3:12b` (量化版本)
- `llama3.1:8b` (量化版本)

v5c使用API版本，不稳定率为0。

**验证方法：**
```bash
# 对比Ollama vs HF版本在同一案例上的行为
python3 scripts/compare_model_versions.py \
  --instability_cases tasks/neutral_stability/instability_cases.jsonl \
  --ollama_models gemma3:12b,llama3.1:8b \
  --hf_models google/gemma-2-9b,meta-llama/Llama-3.1-8B \
  --output tasks/neutral_stability/model_version_comparison.json
```

**假设2: Temperature设置不当**

v7b可能有非零temperature，导致随机性。

**验证方法：**
```bash
# 检查v7b的实际配置
cat configs/factual_perturbation_zh_five_model_v7.json | jq '.models[] | {name, temperature, top_p}'
```

**假设3: API服务波动**

特定时间段的API响应不一致。

**验证方法：**
- 检查不稳定案例的timestamp分布
- 检查retry pattern

**假设4: Prompt敏感性**

某些prompt本身就容易产生不稳定响应。

**验证方法：**
```bash
# 对不稳定的base facts进行多次重复实验
python3 scripts/test_prompt_stability.py \
  --base_facts tasks/neutral_stability/instability_cases.jsonl \
  --model qwen3.6-27b \
  --num_repeats 10 \
  --temperature 0 \
  --output tasks/neutral_stability/prompt_stability_test.json
```

### 3. 优化配置设计 (预计耗时: 2小时)

基于根因分析，设计优化后的配置：

```bash
cat > configs/factual_perturbation_stable_neutral_v1.json << 'CFGEOF'
{
  "experiment_name": "stable-neutral-v1",
  "models": [
    {
      "name": "qwen3.6-27b",
      "provider": "ollama",
      "temperature": 0,
      "top_p": 1,
      "seed": 42
    },
    {
      "name": "bailian/deepseek-v3.2",
      "provider": "api",
      "temperature": 0,
      "top_p": 1
    },
    {
      "name": "gemini-3.7-flash",
      "provider": "api",
      "temperature": 0,
      "top_p": 1
    },
    {
      "name": "google/gemma-2-9b",
      "provider": "hf",
      "device": "cuda",
      "precision": "bf16",
      "temperature": 0,
      "top_p": 1,
      "use_cache": true
    },
    {
      "name": "meta-llama/Llama-3.1-8B",
      "provider": "hf",
      "device": "cuda",
      "precision": "bf16",
      "temperature": 0,
      "top_p": 1,
      "use_cache": true
    }
  ],
  "arms": ["en_original", "en_neutral", "en_targeted", "zh_original", "zh_neutral", "zh_targeted"],
  "neutral_stability_check": {
    "enabled": true,
    "threshold": 2,
    "retry_on_instability": false
  },
  "inference_settings": {
    "batch_size": 1,
    "max_retries": 3,
    "timeout_seconds": 60
  }
}
CFGEOF
```

**关键优化点：**
1. **替换Ollama量化版本为HF原版**
2. **强制temperature=0, top_p=1**
3. **添加seed（对支持的模型）**
4. **HF模型使用bf16精度和KV cache**
5. **禁用retry_on_instability（避免掩盖问题）**

### 4. 验证实验 (预计耗时: 12-16小时)

**在v7b的73条上重跑，验证优化效果：**

```bash
# 准备验证数据（v7b的73条完整候选）
python3 scripts/extract_v7b_candidates.py \
  --three_arm_analysis data_processed/factual_perturbation/zh-mvp-v1/runs/diagnostic-100-five-model-v7b/three_arm_analysis.json \
  --input_bundle data_processed/factual_perturbation/zh-mvp-v1/runs/diagnostic-100-five-model-v7b/input_bundle.jsonl \
  --output tasks/neutral_stability/v7b_73_candidates.jsonl

# 运行验证实验
python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_stable_neutral_v1.json \
  --input tasks/neutral_stability/v7b_73_candidates.jsonl \
  --output_dir data_processed/factual_perturbation/zh-mvp-v1/runs/neutral-stability-verification-v1 \
  --enable_neutral_control
```

**实验规模：**
- 73条 × 6条件 × 5模型 = 2,190个测量点
- 与v7b完全相同的输入
- 成本：~$100 (API部分)

### 5. 结果对比和分析 (预计耗时: 2小时)

```bash
# 对比v7b vs 验证实验的Neutral稳定性
python3 scripts/compare_neutral_stability.py \
  --baseline_results data_processed/factual_perturbation/zh-mvp-v1/runs/diagnostic-100-five-model-v7b \
  --optimized_results data_processed/factual_perturbation/zh-mvp-v1/runs/neutral-stability-verification-v1 \
  --output tasks/reports/neutral_stability_comparison.md
```

**对比指标：**

| 指标 | v7b (baseline) | 优化版 | 目标 |
|---|---:|---:|---|
| Neutral不稳定案例数 | 41/73 | ? | ≤7 (≤10%) |
| PNT候选数 | 14 | ? | ≥14 |
| 平均翻转率 | - | ? | 保持或提升 |

**详细分析：**
1. 41个原不稳定案例中有多少被修复？
2. 是否出现新的不稳定案例？
3. PNT候选数是否保持（确保优化没有降低信号）
4. 每个模型的表现变化

### 6. 生成最终报告和建议 (预计耗时: 2小时)

```bash
python3 scripts/generate_neutral_stability_report.py \
  --instability_analysis tasks/neutral_stability/instability_cases.jsonl \
  --model_comparison tasks/neutral_stability/model_version_comparison.json \
  --verification_results data_processed/factual_perturbation/zh-mvp-v1/runs/neutral-stability-verification-v1 \
  --comparison tasks/reports/neutral_stability_comparison.md \
  --output tasks/reports/neutral_stability_optimization.md
```

**报告内容：**

1. **问题概述**
   - v7b的41个不稳定案例统计
   - 对PNT候选筛选的影响

2. **根因分析**
   - 主要原因（Ollama量化 / temperature / API波动）
   - 次要因素
   - 模型特异性

3. **优化方案**
   - 配置改动清单
   - 理论依据

4. **验证结果**
   - 不稳定率降低幅度
   - PNT候选数保持情况
   - 成本和性能影响

5. **部署建议**
   - 是否在Task 1/2中使用优化配置
   - GPU资源需求（HF模型）
   - 备选方案（如果优化效果有限）

## 质量门控

1. **优化有效性**: Neutral不稳定率从56% (41/73) 降低至≤10% (≤7/73)
2. **信号保持**: PNT候选数≥12条（不能因为优化丢失太多信号）
3. **可复现性**: 在73条上重复运行2次，不稳定率波动<5%
4. **成本可接受**: 验证实验成本≤$150

## 决策标准

**如果优化成功（不稳定率≤10%）：**
- 在Task 1和Task 2中使用优化配置
- 更新所有后续实验的默认配置
- 记录GPU资源需求

**如果优化效果有限（不稳定率10-30%）：**
- 采用宽松的不稳定阈值（threshold=2 → 3）
- 在PNT候选筛选时增加人工复核环节
- 考虑只使用稳定模型的子集（例如只用API模型）

**如果优化失败（不稳定率>30%）：**
- 深入调查是否有数据质量问题
- 考虑调整三臂设计（例如去掉Neutral条件）
- 评估是否改用其他badcase筛选方法

## 输出审核清单

**必须生成的文件：**
- [ ] `tasks/neutral_stability/instability_cases.jsonl`
- [ ] `tasks/neutral_stability/model_version_comparison.json`
- [ ] `configs/factual_perturbation_stable_neutral_v1.json`
- [ ] `data_processed/factual_perturbation/zh-mvp-v1/runs/neutral-stability-verification-v1/experiment_report.md`
- [ ] `tasks/reports/neutral_stability_comparison.md`
- [ ] `tasks/reports/neutral_stability_optimization.md`

**报告必须回答：**
- 41个不稳定案例的根因是什么？
- 优化方案的有效性如何？
- 不稳定率降低了多少？
- PNT候选数是否受影响？
- 是否建议在Task 1/2中使用优化配置？

## 依赖脚本

**需要创建的新脚本：**
- `scripts/extract_neutral_instability_cases.py`
- `scripts/compare_model_versions.py`
- `scripts/test_prompt_stability.py`
- `scripts/extract_v7b_candidates.py`
- `scripts/compare_neutral_stability.py`
- `scripts/generate_neutral_stability_report.py`

**复用的现有脚本：**
- `scripts/run_factual_perturbation.py`
- `scripts/analyze_three_arm_results.py`

## 时间线

- **Day 1**: 不稳定案例分析 + 根因假设验证 (步骤1-2)
- **Day 2**: 优化配置设计 + 准备验证实验 (步骤3)
- **Day 3-4**: 运行验证实验 (步骤4)
- **Day 5**: 结果对比 + 报告生成 (步骤5-6)

**总计: 4-5个工作日**

## 成功标准

✅ 识别41个不稳定案例的根因
✅ 设计并验证优化配置
✅ Neutral不稳定率降低至≤10%
✅ PNT候选数保持≥12条
✅ 提供明确的部署建议

## 后续衔接

**如果优化成功：**
- Task 1使用优化配置
- Task 2使用优化配置
- 更新v7b的14条候选（用优化配置重跑）

**如果优化失败：**
- 评估是否调整三臂设计
- 考虑只使用稳定模型子集
- 增加人工审核环节

## 风险缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| HF模型GPU资源不足 | 中 | 中 | 使用bf16精度，考虑API替代 |
| 优化降低PNT信号 | 低 | 高 | 监控PNT候选数，调整筛选阈值 |
| 根因诊断错误 | 低 | 中 | 多假设并行验证 |
| 成本超预算 | 低 | 低 | 验证实验仅73条，可控 |

## 可选增强

**如果时间和资源允许：**
1. **模型ensembling**: 使用多个模型的majority vote来提高稳定性
2. **Adaptive threshold**: 根据base fact特征动态调整不稳定阈值
3. **Confidence scoring**: 为每个预测添加置信度评分，过滤低置信度案例
4. **Extended validation**: 在Task 1的50条上也验证优化效果

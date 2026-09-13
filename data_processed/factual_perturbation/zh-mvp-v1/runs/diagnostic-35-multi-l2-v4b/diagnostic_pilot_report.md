# 100 条诊断性中文 pilot 报告

## 结论

本轮证明了端到端链路可运行，也观察到中文 Targeted 条件下的目标 distractor 特异性翻转；但证据尚不足以冻结候选或调用 holdout。

- 冻结输入为 100 个独立 base triple，最终 Codex 代理审核接受 35 个独立 source，各保留 1 个 L2 候选。
- 四个 Simulation 模型完成 840/840 个结果，35/35 个候选具有完整的 24 个结果。
- 6/35 个候选至少出现一次中文目标 distractor 特异性翻转，共 10 个模型×候选事件；non-target Targeted error 为 0。
- 只有 1/35 个候选同时满足英文 Targeted 全对、中文 Original/Neutral 全对且中文 Targeted 命中目标 distractor的严格离线筛选条件。
- 这 1 个严格候选由 `bailian/deepseek-v3.2` 触发，并且是 Codex 对自动 perturbation judge 的覆盖项；它不是 human gold，也没有经过权威外部事实核验。
- 中文特异翻转事件为 10/140，英文为 6/140；配对 exact McNemar 检验 `p=0.21875`，不能据此宣称存在显著的中文特异效应。

因此，本轮结果属于“链路成功且出现弱信号”，不是“实验假设已得到确认”。应先盲态复核和独立重复候选，再决定是否冻结或预注册其他语言。

## 设计与隔离

- 权威 run：`diagnostic-35-multi-l2-v4b`
- 输入 run：`diagnostic-100-multi-upstream-v4`
- 独立样本单位：`source_id` / base triple
- endpoint：三选项多选
- arms：Original、matched Neutral、Targeted
- languages：English、Chinese
- Simulation：`qwen3.6-27b`、`bailian/deepseek-v3.2`、`gpt-5-mini`、`gemini-3.7-flash`
- `Paths Not Taken=false`
- `holdout=false`
- 未调用 `claude-opus-4-8`
- 未调用 SenseNova
- `reviewer_type=codex_proxy`
- `not_human_gold=true`

`paths_not_taken_candidate` 只是离线分析标签，不表示启用了 Paths Not Taken 或调用了 holdout。

## 数据漏斗

| 阶段 | 数量 | 备注 |
|---|---:|---|
| 冻结 base triples | 100 sources | 与先前 50 条 source 不重叠 |
| 翻译完成 | 100 sources | 100% |
| 翻译自动复核接受 | 82 sources | 82% |
| distractor 自动接受 | 126 candidates / 74 sources | 每个 source 最多两个原始错误选项 |
| perturbation generation 完成 | 113 groups / 72 sources | 13 groups、12 sources 永久失败并保留 |
| perturbation 自动 judge 接受 | 87 candidates / 52 sources | 自动审核分支 |
| 三选项结构可用并进入 Codex 审核 | 92 L2 candidates / 51 sources | 至少两个接受的 distractor |
| Codex 代理审核接受 | 35 candidates / 35 sources | 其中 9 个覆盖自动 reject |
| 完整 Simulation | 35 candidates / 35 sources | 840/840 outcomes |

最终分析样本是 35 个独立 base triple，而不是 100 个。输入到分析的保留率为 35%。

## Simulation 可靠性

- 物理调用：840
- 完成：840
- 累计尝试：851
- 重试：11
- 最终失败：0
- 首轮有 1 个 `gpt-5-mini` 结果在 4 次结构校验失败后成为终态失败；断点续跑又尝试 3 次并成功。
- Simulation tokens：116,678
- Simulation 平均/最大延迟：3,581.09 / 29,737 ms
- 全链路 tokens：487,405
- 全链路最终失败：13，全部为上游 perturbation generation group，不属于 Simulation。
- 费用无法审计：未配置版本化 provider price table；token 用量已保留，可与账单对账。

| 模型 | 完成 | 重试 | 最终失败 | 平均延迟 ms | 最大延迟 ms |
|---|---:|---:|---:|---:|---:|
| `qwen3.6-27b` | 210/210 | 0 | 0 | 609.27 | 3,617 |
| `bailian/deepseek-v3.2` | 210/210 | 0 | 0 | 882.46 | 1,588 |
| `gemini-3.7-flash` | 210/210 | 2 | 0 | 5,098.26 | 15,903 |
| `gpt-5-mini` | 210/210 | 9 | 0 | 7,734.38 | 29,737 |

## 三臂结果

每格分母为 35 个候选。

| 模型 | EN Original | EN Neutral | EN Targeted | ZH Original | ZH Neutral | ZH Targeted |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b` | 33/35 | 33/35 | 31/35 | 32/35 | 32/35 | 30/35 |
| `bailian/deepseek-v3.2` | 35/35 | 34/35 | 32/35 | 35/35 | 34/35 | 29/35 |
| `gpt-5-mini` | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 |
| `gemini-3.7-flash` | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 | 34/35 |
| 合计 | 138/140 (98.6%) | 137/140 (97.9%) | 133/140 (95.0%) | 137/140 (97.9%) | 136/140 (97.1%) | 128/140 (91.4%) |

Targeted 相对 Neutral 的准确率下降：English 2.9 个百分点，Chinese 5.7 个百分点；描述性差中差为 2.9 个百分点。样本量和配对检验均不支持将该差异解释为已确认的语言效应。

## 特异性与对照

- 中文 stable-baseline 单位：136/140。
- 中文 Targeted 目标 distractor 特异性翻转：10 个模型×候选事件，分布在 6 个候选。
- 英文 Targeted 目标 distractor 特异性翻转：6 个模型×候选事件，分布在 4 个候选。
- 中英文配对：两种语言都翻转 5、仅英文翻转 1、仅中文翻转 5、两者都不翻转 129。
- 中文 non-target Targeted error：0。
- Neutral-control instability：2/280 个模型×候选×语言对，均来自 `bailian/deepseek-v3.2`。
- shared-baseline inconsistency：0。

中文翻转按模型分布：

| 模型 | 特异性翻转事件 |
|---|---:|
| `bailian/deepseek-v3.2` | 6 |
| `qwen3.6-27b` | 3 |
| `gemini-3.7-flash` | 1 |
| `gpt-5-mini` | 0 |

严格离线筛选候选为 `raw_sciq_000351_source_wrong_2_p2`。其问题事实为“moraine 由 glacier 沉积”，目标 distractor 为 `wind`；只有 `bailian/deepseek-v3.2` 在中文 Targeted arm 从正确答案翻到目标 distractor，其他模型以及所有英文 arms 均保持正确。

该候选的自动 perturbation judge 原决定为 reject，失败项为 `relation_preserved`、`target_distractor_salience` 和 `strength_appropriate`；Codex 代理审核覆盖为 accept。这个 judge disagreement 是当前最重要的不确定性。

## 统计解释

- 严格候选率：1/35 = 2.86%。
- 95% Wilson 区间约为 0.51%–14.53%，区间很宽。
- 100 是输入分母，35 才是分析分母；不能将 1/35 写成 1/100 的效应率。
- 本轮候选经过强筛选且 9/35 为自动 judge override，因此结果不能直接外推到原始数据总体。
- 四个 Simulation 模型不是独立同分布的人类受试者；模型×候选事件不能当作 140 个独立 base triple。

## 工程审计

首次创建的 `diagnostic-35-multi-l2-v4` 暴露出一个断点语义缺陷：代理审核 rerun 会重试复用上游中的永久失败 generation group。该进程已中止且 run 被保留审计，没有作为结果来源。

修复后：

- 代理审核 rerun 启动前校验复用上游 artifact 的 SHA-256。
- 翻译、翻译复核、distractor 审核、generation 和 perturbation 审核的已有终态被冻结，不会在 Simulation rerun 中重跑。
- Simulation 自身仍允许对失败项断点续跑。
- 回归测试 20/20 通过。

## 决策建议

当前不应调用 holdout，也不应把 1 条候选当作论文级确认结果。优先顺序应为：

1. 对 6 条中文翻转候选做再次盲态 Codex 复核，特别审查事实、三选项唯一性、英文/中文等价性、target salience 和 neutral matching；严格候选必须单独解决自动 judge 与 Codex 的分歧。
2. 对通过复核的候选做独立重复调用，预先固定重复次数和成功判据，确认翻转不是 provider 随机性或单次解码波动。
3. 若严格信号仍只有约 1 条，不继续无界加强扰动；应预注册另一种语言作为比较，并保留完全相同的三臂、多选、筛选和模型方案。
4. 只有在候选经复核、重复和冻结后，才考虑调用 `claude-opus-4-8` holdout。

本轮无需立即放弃中文：链路已有非零信号。但它支持的是继续做候选级确认，而不是直接扩大中文样本或开启 Paths Not Taken。

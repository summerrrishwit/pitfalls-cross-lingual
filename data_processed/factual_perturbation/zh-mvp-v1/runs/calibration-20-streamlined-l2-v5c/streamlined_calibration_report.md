# Streamlined v5 中文校准报告

## 结论

streamlined v5 已完成一轮与历史 150 条 source 零重叠的 20 条中文校准。工程链路和预设质量阈值均通过，且出现了非零的中文特异信号；但样本仍只够决定“可以进入新的 100 条 v5 诊断 pilot”，不足以确认跨语言效应或冻结候选。

- 20 条输入中，翻译最终接受 18 条（90%），恰好达到门槛。
- 17/20 个 source 具备完整三选项结构（85%），超过 80% 门槛。
- 35/35 个 perturbation generation group 完成（100%），无永久失败。
- Codex 盲审保留 15 个独立 source，每个 source 仅 1 个候选；最终保留率为 15/20（75%），超过 50% 门槛。
- 四个 Simulation 模型完成 360/360 个逻辑测试单元，15/15 个候选均有完整的 24 个结果。
- Neutral-control instability 为 0/120（0%），低于 2% 门槛；shared-baseline inconsistency 为 0；Targeted non-target error 为 0。
- 3/15 个候选出现至少一次中文目标 distractor 翻转，共 6/60 个模型×候选事件；其中 2/15 个候选满足严格条件。
- `Paths Not Taken` 和 holdout 均保持禁用；未调用 `claude-opus-4-8` 或 SenseNova。

`three_arm_analysis.json` 中的 `paths_not_taken_candidate=true` 只是离线筛选标签，不表示已启用 Paths Not Taken。

## 运行与隔离

- 上游 run：`calibration-20-streamlined-upstream-v5b`
- Simulation run：`calibration-20-streamlined-l2-v5c`
- endpoint：三选项多选
- arms：Original、matched Neutral、Targeted
- languages：English、Chinese
- Simulation：`qwen3.6-27b`、`bailian/deepseek-v3.2`、`gpt-5-mini`、`gemini-3.7-flash`
- 审核：`reviewer_type=codex_proxy`、`not_human_gold=true`、`review_blinded_to_simulation_results=true`
- 审核包 SHA-256：`08521dcd513181131e7478a689c3964d9550c7355777c4e63f419ea47cb0f6ba`
- Codex 审核接受的强度分布：4 个 `l2_lexical`、11 个 `l2_relational`

## 数据漏斗

| 阶段 | 数量 | 比率或说明 |
|---|---:|---|
| 冻结输入 | 20 sources | 与历史 150 条 source 零重叠 |
| 翻译完成 | 20 | 100% |
| 翻译复核最终接受 | 18 | 90% |
| distractor 审核 | 56 candidates | 接受 41 |
| 完整三选项 source | 17 | 85% of input |
| perturbation generation | 35/35 groups | 100%，3 次重试 |
| 进入盲审包 | 68 candidates / 17 sources | 自动 judge 仅作建议 |
| Codex 盲审接受 | 15 candidates / 15 sources | 75% of input；每 source 最多 1 条 |
| 完整 Simulation | 15 candidates | 360/360 逻辑测试单元 |

相较旧 v4 的 100-input diagnostic pilot，最终分析保留率从 35/100（35%）提高到 15/20（75%）。两批样本独立但流程和样本量不同，因此 Fisher exact `p=0.00121` 只能作为“简化流程减少样本流失”的探索性证据，不能单独归因于某一个改动。

## 三臂结果

每格分母为 15 个候选。

| 模型 | EN Original | EN Neutral | EN Targeted | ZH Original | ZH Neutral | ZH Targeted |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b` | 14/15 | 14/15 | 14/15 | 14/15 | 14/15 | 12/15 |
| `bailian/deepseek-v3.2` | 14/15 | 14/15 | 14/15 | 14/15 | 14/15 | 12/15 |
| `gpt-5-mini` | 14/15 | 14/15 | 13/15 | 14/15 | 14/15 | 13/15 |
| `gemini-3.7-flash` | 14/15 | 14/15 | 14/15 | 14/15 | 14/15 | 13/15 |
| 合计 | 56/60 | 56/60 | 55/60 | 56/60 | 56/60 | 50/60 |

Targeted 相对 Neutral 的准确率下降为：English 1.67 个百分点，Chinese 10.00 个百分点，描述性差中差为 8.33 个百分点。

中文目标 distractor 翻转共有 6/60 个模型×候选事件，英文为 2/60。配对后，两种语言都翻转 2、仅中文翻转 4、仅英文翻转 0、两者都不翻转 54；exact McNemar `p=0.125`，尚不能宣称存在统计显著的中文特异效应。

所有 6 个中文翻转事件都来自 `l2_relational`，4 个 `l2_lexical` 候选未出现翻转。由于 Codex 是按质量择优而非随机或平衡分配强度，这只能说明当前 relational 方案更有信号，不能据此估计两种机制的因果差异。

## 候选解释

严格通过的两条候选为：

1. `raw_mmlu_000016_source_wrong_0_p2`：问题答案为 `Dysprosody/韵律障碍`，目标 distractor 为 `Dysarthria/构音障碍`。仅 `gemini-3.7-flash` 在中文 Targeted 从正确答案翻到目标 distractor；所有英文 arms 和中文 Original/Neutral 均正确。
2. `raw_sciq_000103_source_wrong_0_p2`：问题答案为 `photosynthesis/光合作用`，目标 distractor 为 `digestion/消化`。`qwen3.6-27b` 与 `bailian/deepseek-v3.2` 在中文 Targeted 翻到目标 distractor；所有英文 arms 和中文 Original/Neutral 均正确。

另有 `raw_ai2_arc_easy_000428_source_wrong_0_p2` 在中文产生 3 个目标翻转，但其中 `bailian/deepseek-v3.2` 和 `gpt-5-mini` 在英文 Targeted 也翻转，因此它证明扰动足够强，却不满足跨语言特异性控制。

严格候选率为 2/15（13.33%，95% Wilson 区间约 3.74%–37.88%）。旧 v4 pilot 为 1/35（2.86%，约 0.51%–14.53%）；两者 Fisher exact `p=0.211`，不能认为严格候选率已显著提高。

## 稳定性、断点恢复与资源

- Simulation 逻辑测试单元：360；最终完成：360；provider attempts：391；重试：31；永久失败：0。
- `qwen3.6-27b`：90/90 完成，0 重试，最终记录平均/最大延迟 652.86/1,854 ms。
- `bailian/deepseek-v3.2`：90/90 完成，0 重试，891.52/1,697 ms。
- `gemini-3.7-flash`：90/90 完成，6 重试，5,756.29/20,229 ms。
- `gpt-5-mini`：90/90 完成，25 重试，7,557.21/37,883 ms。
- 首轮为 357/360；第一次断点续跑变为 358/360。两条长英文电影上下文在 256-token completion 预算下持续返回空内容，预算提高到 1,024 后仅重跑失败项并达到 360/360。
- 完成后再次执行同一 run，`redacted_events.jsonl` 行数保持 365 不变，证明已完成项不会被重复调用。
- 按不重复的上游辅助 checkpoint 加 Simulation 重算，全链路共有 574 个逻辑记录、608 次 provider attempts、34 次重试、0 个最终失败，合计 190,436 tokens；其中 Simulation 为 51,852 tokens。
- 现有 `preholdout_summary.json` 的 184,752 tokens 未纳入部分翻译修复和 distractor 补生成辅助 checkpoint，因此最终资源审计应采用本报告的 190,436-token 口径。
- 代码库没有版本化 provider 价格表，因此不能给出可审计的货币费用。正式 gate 仍显示 `currency_cost_not_auditable`，但模型、tokens、重试和延迟均已记录。
- 579 条上游与 Simulation 事件日志均标记 `credentials_or_endpoints_included=false`。

## 与实验设计预期的关系

符合预期的部分：减少硬过滤后保留率明显提升；定向修订、补生成、双 L2、Codex 盲审、三选项 endpoint、matched Neutral、四模型覆盖和断点续跑均工作；Neutral 与 non-target 控制干净；中文特异信号不再为零。

仍有出入的部分：翻译接受率只刚好到 90%，没有安全余量；严格效应只有 2 条且区间很宽；`l2_relational` 与 `l2_lexical` 数量不平衡；`gpt-5-mini` 对长提示的结构化输出需要更大预算；货币成本仍不可审计。

## 决策

建议按同一 streamlined v5 设计运行新的 100-input-base 中文诊断 pilot，但不要立即冻结候选或调用 holdout。100 条 run 应预注册：

1. 输入分母固定为 100 个新 source，并报告每层流失；不把 perturbation 数量当独立样本。
2. 保持三选项、Original/Neutral/Targeted、中英双语和当前四模型不变。
3. 保持 Codex 在 Simulation 前盲审、每 source 最多 1 个候选；自动 perturbation judge 继续仅作建议。
4. 主终点仍是英文 Targeted 全对、中文 Original/Neutral 全对且中文 Targeted 命中目标 distractor；跨语言共同翻转单独报告，不能计入主终点。
5. 不再整体增强扰动。当前证据显示 blanket strengthening 会产生语言无关效应；若需比较两种 L2 机制，应预先设定平衡子样本，而不是事后挑选。
6. 在新 100 条 v5 pilot 完成复核与重复确认前，继续保持 `Paths Not Taken=false`、`holdout=false`。

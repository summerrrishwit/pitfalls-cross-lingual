# Qwen3-8B 160 条实验：远端恢复与必要分析

2026-10-16，全部模型推理、tokenizer 审计、代码测试、完整答案序列评分、阳性复跑与干预均在远端完成。本地仅准备代码、同步材料并保存结果副本。

最新状态（2026-10-17）：v5b通过G2并完成规则固定后的Validation32facts独立确认，但G3未通过：10个定向缺陷facts/18variants修复为0，实际生成增益0。Sealed0/32保持未开启；完整性核验通过，推理任务结束。见 [v5b最终结果](repair_v5_outcome.md)。此前v3/v4两轮8候选未通过G2的结论继续独立保留，见 [旧轮修复结果](repair_outcome.md)。

## 运行位置

- SSH：`root@connect.cqa1.seetacloud.com:24946`
- 权威目录：`/root/autodl-tmp/pnt-open-development-20261016`
- Python：`/root/autodl-tmp/venvs/vllm/bin/python`
- 模型：`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，原冻结 BF16/eager/deterministic 设置。
- 原冻结数据、checkpoint 文件、G0B 模型与渲染记录均通过 SHA 检查。未调用 API 模型或本地模型。

## 已完成

1. 远端执行检查 10/10 通过；tokenizer 审计通过 960 个开放三臂输入、1,920 个 vector-source prompts、26 个 controls，最长 253 prompt tokens。
2. Development 全部 96 facts、双 variant、EN/ZH Original/Neutral/Targeted 共 960/960 输入完成。192 个 Original 生成在 exact render、旧结果 SHA 与 8 个按哈希选定的 fresh generation 检查后复用；768 个 context 生成新跑。全 960 输入重新评分所有冻结答案组的全部语言内 aliases。
3. 完成 26/26 无关事实 no-op baseline；其语义复核为 15 正确、8 错误、3 defer，后续保持评估不能把 defer 当成错误。
4. 对 516 个需要复核的唯一 cohort 输出进行显式 Codex proxy 裁决；15 个输出级判断保留为缺失值，无未审输出。判断绑定 generation SHA，`human_gold=false`。
5. 对所有 25 个中文阳性 variant 的三臂及 5 个严格案例的英文三臂去重后进行独立复跑：82/82 生成文本一致，覆盖 17 facts。这是同一确定性环境下的复现，未检验跨硬件/随机采样稳定性。

## 必要结果（按独立事实计数）

| 实际生成终点 | 阳性 | 阴性 | 缺失 |
|---|---:|---:|---:|
| 中文 Original/Neutral 正确，Targeted 命中指定错误答案 | 17 | 74 | 5 |
| 上述成立且英文三臂全部正确 | 5 | 89 | 2 |

17 个定向缺陷事实对应 25 个 variants；variant 不能作为独立样本。

| MCQ 标签 | open 定向阳性 | open 阴性 | open 缺失 |
|---|---:|---:|---:|
| MCQ 阳性（57 facts） | 12 | 41 | 4 |
| MCQ 阴性（39 facts） | 5 | 33 | 1 |

因此后续可以围绕已复现的 open 缺陷推进机制与修复，不能直接把原 57 个 MCQ 阳性视为 open 缺陷；MCQ 阴性中存在额外 open 阳性。

候选完整答案序列偏好迁移为 57 facts，只作辅助终点，与实际生成阳性 17 facts 分开报告。没有生成长度耗尽或技术失败。基线失败、Neutral 不稳定、非定向错误和缺失状态保留在逐条标签中，不回填为负例。

## 后续修复结果

已在同一远端收集 960 条开放输入的 36 层 resid_pre 轨迹及 1,440 条四折向量来源，每折使用另外 72 个 Development facts。4 个 no-op hook 检查生成完全一致，完整序列 margin 差异均小于 5e-9。

v3 完成 12 设置完整序列辅助评分和 3×532 次实际生成；v4 另行预先固定较小强度的 5 配置，完成 5×532 次实际生成。两轮共 4,256 次候选生成，每个配置覆盖全部 480 个中文 cohort 输入和 4 折×13 个中文无关对照。8 个候选均违反保留条件。失败候选未审输出与仍不明确的日期等回答继续保留缺失，不回填负例。

G2 未通过，按事先写入配置的 futility 条款停止；随机比较仅为合格候选运行，因此本次未启用。最终全 Development 向量、修复规则冻结、Validation 和 Sealed 均未执行。轨迹比较是首答案 token 的描述性结果，不构成检索/翻译阶段的因果证明。

远端独立结果核验通过：逐条输出与语义判断 SHA、完整计数、四折来源隔离、已知保留损伤重新计数、no-op 一致性及留出集未启用状态均一致。本地同步的 40 个绑定证据文件再次通过 SHA 校验。结果归档固定不等于修复规则通过或独立确认完成。

## 证据文件

- [完整汇总](remote_artifacts/results/bridge_summary.json)
- [事实级标签](remote_artifacts/results/open_fact_labels.jsonl)
- [variant 标签](remote_artifacts/results/open_variant_labels.jsonl)
- [逐条生成及标签](remote_artifacts/results/open_results_labeled.jsonl)
- [显式语义裁决](remote_artifacts/results/semantic_review_decisions.jsonl)
- [完整性审计](remote_artifacts/results/integrity_audit.json)
- [阳性复跑](remote_artifacts/results/positive-repeat/reproduction_summary.json)
- [原运行配置](run_config.json)

运行配置与数据输入已固定在首次模型输出之前。开放生成主终点、序列偏好辅助终点替代旧 G2A 的 OR 合并口径；本轮为 Development 方法开发，未修改或覆盖旧冻结结果。

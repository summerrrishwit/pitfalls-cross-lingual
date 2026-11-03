# 远端后续实验结果：G2 未通过，独立确认未开启

本轮授权范围为继续现有 Qwen3-8B 160-fact 实验至规则固定及独立确认。实际完成了必要行为复现、层轨迹与两轮有界修复；**8 个候选均未通过预先固定的保留条件，没有可冻结的最终修复规则**。因此尚未达到独立确认阶段，不能报告修复成功。全部模型计算和执行检查均在远端完成。

## 已完成的必要步骤

- 开放三臂：Development 96 facts、960 输入；17 个定向缺陷 facts，25 variants。阳性三臂复跑 82/82 文本一致。
- 机制材料：960 个开放输入的 36 层 resid_pre 轨迹，1,440 条四折向量来源；每折使用另外 72 facts，查询与示例来源均隔离评估折。
- hook 等价性：4/4 no-op 生成完全一致，完整序列分数误差 <5e-9。
- v3：12 设置的完整答案序列辅助评分，随后 task / difference / combined 三配置各 532 次生成。
- v4：另行预先固定较小强度的五配置，直接以实际生成选择，各 532 次生成；未改变向量估计器、分折或保留门槛。
- 两轮共 **4,256 次候选生成**；每配置 480 个中文 cohort 输入和四折共 52 个中文无关事实对照。英文干预触发关闭，保留既有英文基线。

## 为什么不能固定规则

抗扰动保留面板共 14 facts，预先规定损伤率最多 5%；已知伤害 1 fact 即为 7.14%，不能通过。缺失判断不按错误或正确处理。

| 候选 | 已知受损保留 facts（至少） | 结论 |
|---|---:|---|
| v3 task / difference / combined | 5 / 2 / 4 | 均不通过；task/combined 另有无关事实伤害 |
| v4 task，layer 12，+0.25 | 1 | 不通过 |
| v4 task，layer 12，+0.5 | 2 | 不通过 |
| v4 difference，layer 24，−0.25 | 2 | 不通过 |
| v4 difference，layer 24，−0.5 | 1 | 不通过 |
| v4 combined，+0.25 / −0.25 | 2 | 不通过 |

具体损伤包括把原本正确的玛格特·罗比改成其他演员，把逆转录转座子改成 DNA 修复蛋白复合物。最弱 task 配置也把一手数据的“专门首次收集”定义替换为“使用前已收集、未分析”；后者缺失一手来源/专门首次收集的核心含义。

最弱配置原有三个主要判断缺失。完整答案组和基线一致性复核明确了上述定义错误及一个风险人群正确释义，修订前记录、修订理由与文本 SHA 均保留；只回答“2003年”的日期输出仍为缺失。该配置已有明确保留损伤，剩余日期判断无法使它通过。其余失败配置不再做无助于门槛决定的全量语义裁决，163 个尚未裁决的唯一输出保留为未审，不伪造负例。全部裁决为 Codex proxy，`human_gold=false`。

辅助 margin 增益和未完全判定子集上的准确率变化不能替代保留检查，也不能作为整体成功率。首答案 token 轨迹仅提供描述性对比，未证明检索/翻译阶段的因果机制。

## 独立确认状态

最终 gate 为 `G2_failed_no_eligible_generation_candidate`。配置事先规定：无 Development 合格候选就停止，不消耗后续留出数据。

- 合格候选为 0，因此未启动五个随机方向比较。
- 未收集全 96 facts 的最终来源向量，未生成固定修复规则。
- **Validation=0/32，Sealed=0/32**；未产生任何独立分区 opened 记录。
- 独立确认代码已准备；远端入口测试确认失败与证据不足状态均无法绕过 G2 生成规则。

后续若继续，需要先制定新的 Development 修复方案并通过保留门槛；通过后才依次做随机方向比较、固定最终向量及规则、Validation、Sealed。当前有限候选池的结果不支持直接进入独立确认。

## 证据与核验

远端权威目录：`/root/autodl-tmp/pnt-open-development-20261016`。两轮推理任务退出码均为 0，当前没有遗留推理任务。

独立于选择分析器的保存结果核验通过：8 配置逐条计数与唯一性、四折来源、向量 SHA、原始/标注文本一致性、裁决文本绑定、保留损伤重新计数、no-op 与未开启留出集状态。同步后的 40 个绑定证据文件校验通过。这里只是实验完整性核验，不是留出集独立确认。

- [最终决定](remote_artifacts/repair_final_decision.json)
- [完整性核验与文件 SHA](remote_artifacts/repair_closure_audit.json)
- [v4 门槛结果](remote_artifacts/repair-v4/generation_selection.json)
- [v4 原始指标与缺失数](remote_artifacts/repair-v4/generation_metrics.json)
- [v4 显式语义判定及修订](remote_artifacts/repair-v4/semantic_review_decisions.jsonl)
- [v3 门槛结果](remote_artifacts/repair-v3/generation_selection.json)
- [预先固定的 v4 配置](repair_config_v4.json)

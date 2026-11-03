# V5b 修复及固定后独立确认结果

2026-10-17。本轮已完成局部恢复诊断、新修复候选、随机/时机/来源数量对照、最终规则冻结和一次完整 Validation。**Development 通过 G2，但固定规则在 Validation 未通过 G3；没有获得可泛化修复的证据。Sealed 32 条保持未开启。** 全部推理、评分、测试和分析执行均在远端完成，本地保存代码和结果副本。

## 结果

| 阶段 | 实际观察 | 结论边界 |
|---|---|---|
| 同事实局部诊断 | 层28恢复11/25阳性variants，反向使9个变错 | 说明局部可恢复；不是可迁移修复 |
| 完整 Development | 仅3%成对 prefill方向通过生成/保留门槛，修复1个阳性variant | 开发选择结果，效果很小 |
| 五随机方向 | 5组定向实际生成增益均为0，候选高于随机均值 | G2通过 |
| 固定后 Validation | 32 facts全执行，10个定向缺陷facts/18 variants，修复0个 | G3未通过 |
| Sealed | 0/32 | 按预先固定的停止条件保留 |

Development 唯一新增正确案例是“美国队长”的演员由雅尼克·比森变回克里斯·埃文斯，仅覆盖该事实两个变体中的一个。事实等权的定向生成增益为 +2.94 个百分点，全体可评估 Targeted facts 增益约 +0.54 个百分点。候选层位由整个 Development 的诊断选择；四折仅隔离方向及范数来源，这不是完全嵌套交叉验证的无偏估计。

最终固定规则：zero-based layer28，训练来源 Neutral 状态参考范数的3%，同事实 Neutral−Targeted 的事实等权方向，只施加于最后 prompt token。全96个 Development facts 汇总最终向量；480个最终ICL来源及其donor均验证仅在 Development。14个代码文件、配置、最终向量、模型/渲染及评分缓存、Development裁决和指标均绑定SHA，固定后未依据Validation结果调整。

## 独立确认为什么未通过

Validation 完成320个开放no-op输入、320个MCQ评分、173个修复输入以及5×173个随机对照输入，合计1,358条生成记录。MCQ仅用于表型记录。

- 18个已知定向缺陷variants中，恢复正确为 **0**；全体中文Targeted也没有新增正确或新增错误输入。
- 全体及定向子集的事实等权实际生成增益均为 **0**，与五随机方向均值0相同。
- 全体Targeted完整答案相对分数间隔改善约 **+0.783**，定向子集约+0.931，但辅助分数改善没有转化为正确生成。
- 预定的层28描述性归因对比条件也未满足。该首答案token对比本身不构成独立因果证明。
- Original/Neutral配对准确率未下降，2个可评估保留facts和原正确无关对照未发现新伤害；这个面板过小，不能据此证明低损伤泛化。

正式状态为 `G3_validation_failed_or_insufficient`。没有触发“阳性过少”条件：可判定的定向缺陷为10 facts，超过预定最低4个；本次未通过的核心是实际生成没有改善和未高于随机均值。

174个唯一新输出已全部显式Codex proxy复核，`human_gold=false`；其中5个语义判断保留缺失，没有未审输出。1个Original中英文人名冲突事实从主要配对口径排除，主要准确率比较使用31 facts，原始32 facts全部保留。冻结分析器的bootstrap区间[0,0]来自观测配对差全部为0，不能解释为总体效应被精确证明等于0。

## 数据量与修复方式

18、36来源facts各两个预定种子，与72来源的实际定向生成增益均为 .0294118，保留损伤0。这是在固定113输入/25facts诊断面板上的小规模来源检查，没有观察到来源数量增加带来生成改善，不能排除更大样本或其他方法的数据量效应。

同一成对方向在每个答案token施加时，固定诊断面板的定向增益为 .0882353；反方向增益0且伤害1个保留fact。这些对照未临时加入候选池，也未执行全480输入验证。现有证据不支持把旧轮失败简单归因于持续注入，更不能仅靠扩量宣称能修复。

本轮结束于预先约定的G3停止条件。若继续，需要新的Development修复设计，并在开启任何后续确认前预先固定独立确认方案；已开启的Validation不能反复充当新规则的独立确认集。

## 完整性与证据

远端权威目录：`/root/autodl-tmp/pnt-open-development-20261016/repair-v5b`。Development主流程5,818条生成记录，另有自身增量预检；Validation 1,358条生成、320条MCQ评分，任务退出码0，推理任务已结束。7项相关远端检查通过。

独立于确认分析器的完整性检查通过：14个冻结代码文件、固定缓存/向量、32个分配事实、输出计数与唯一性、全文不变及裁决SHA、原/标注记录、Sealed未开启状态。同步到本地的20个Validation绑定文件和3个终局绑定再次校验通过，激活张量继续保存在远端。

- [终局决定](remote_artifacts/repair-v5b/final_decision.json)
- [固定规则清单](remote_artifacts/repair-v5b/repair_rule_frozen.json)
- [Validation正式确认结果](remote_artifacts/repair-v5b/validation/confirmation.json)
- [独立确认完整性核验](remote_artifacts/repair-v5b/validation/confirmation_integrity_audit.json)
- [实际修复/损伤事件数](remote_artifacts/repair-v5b/validation/event_counts.json)
- [Validation显式裁决及缺失理由](remote_artifacts/repair-v5b/validation/semantic_review_decisions.jsonl)
- [G2随机比较](remote_artifacts/repair-v5b/selected_rule.json)
- [来源数量/时机诊断](remote_artifacts/repair-v5b/auxiliary_diagnostic_summary.json)
- [v5b配置](repair_config_v5b.json)、[预先固定协议与预检修正](repair_v5_contract.md)
- [保留的v3/v4失败结果](repair_outcome.md)

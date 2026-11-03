# 开放事实召回缺陷生成：有限反馈与效率评估

更新：2026-10-20。此文是**新候选构建的活动入口**，替代“先挖掘 MCQ 定向弱点再准入 PNT”的新实验流程。
既有160条与已执行实验按其冻结版本解释，不回写旧标签；原始QA的错误选项仍可作为干扰素材。
本次完成协议、离线分析与预算核验支持，不启动任何新模型调用，也不把旧160专用HF运行器声称为通用新生成器。

## 1. 目标和输入

目标：单位预算生成更多**独立、有效、可复现的开放事实召回缺陷**，再用于机制分析、归因和修复。
真实误导、直接错误断言等均可，保存 `manipulation_family`；不把机制尚未解释作为行为候选排除理由。

输入必须有来源QA、独立确定的正确答案/别名、开放问题、base_fact_id、split/component与版本。
source wrong option 是优先干扰素材，必须确认对改写后的开放问题仍然错误。原始选择题没有错误选项时，
可用独立审核的同答案类型候选；不能把干扰答案写回 canonical answer。

```text
来源QA → 事实与正确答案 → 无选项问题 → Original开放基线
                                       ↓
已审核错误选项/候选 → 一个Targeted + 配对Neutral → 开放三臂测量
                                       ↓
终审与阳性复跑 → 缺陷及对照冻结 → 机制归因和修复 → 独立确认
```

MCQ答题、第三选项foil、MCQ阳性标签和多代理投票均不是准入要求。可选MCQ仅作辅助对比，调用开销照计。

## 2. 直接开放测量

- 模型实际生成答案；按冻结别名规则与必要语义审核判分。序列概率偏好单列，不代替正确生成。
- 同一语言 Original 正确、Neutral 正确、Targeted 出现明确事实错误，标为开放诱导缺陷。
- 若Targeted生成指定错误答案，再标为定向缺陷；其他明确事实错误标为非定向缺陷。
- 英文三臂均正确定义更窄的目标语特异层，不作为一般目标语缺陷的硬性要求。
- Original错误、Neutral不稳定、拒答、格式错误、技术失败与语义未确定分别报告；未确定不是负例。
- 对直接错误断言，统一题意仍询问现实事实；若题意改成“按背景作答”，不能沿用原缺陷定义。

每事实首轮一个攻击目标、一对Neutral/Targeted；英中成对共6个输入。五语扩展与英语共六语时为18个，
Natural就是对应Original，不额外重复生成。英语输出跨语言比较共享。复用要求模型/render/评分合同一致。

## 3. 只允许一次反馈修订

默认策略见 [`open_factual_discovery_v1.json`](../configs/open_factual_discovery_v1.json)：

| 上限 | 默认值 |
|---|---:|
| 首轮Development事实 | 32 |
| 每事实语义候选总数 | 2（首次1个，可选修订1个） |
| 每事实反馈修订次数 | 1 |
| 全局生成策略版本 | 2（原版+一次汇总修订） |
| 本轮所有模型物理请求尝试 | 1,024 |
| 总token预算 | 1,000,000 |

这些是可配置的工程上限，不是已批准的推理预算或预期缺陷数量。实际执行前固定成员清单、策略、
模型、预算与运行目录；不得为了达到阳性数量临时追加范围或自动提高上限。

1. 固定首轮策略，每事实生成一个短上下文和匹配Neutral，独立审核后测开放回答。
2. 首轮结束后只汇总一次失败类别。成功事实不再改写；基线失败不进入扰动修订。
3. 对已明确失败且可修订的Development事实，最多一次定向修订，仍是一个候选；同步重审Neutral匹配性。
4. 成功且复现通过即停止；第二次仍失败、预算到限或无可修订原因均停止，保留为失败或对照。

不做beam search、逐条无限反馈、累加旧context、第三轮“再试一次”或根据效果反复切模型。
技术重试保持同一语义候选身份，另记物理请求；pending/技术失败/未裁决不能被当作语义失败触发新候选。
候选修订使用新的candidate_id、输入hash与运行目录，旧输入及结果不可覆盖。
开发反馈只修改生成策略/上下文，不修改事实真值、答案、评分规则或审核质量标签。

## 4. 统计预算与独立缺陷产出

全部来源事实和尝试留在账本中；原始QA没有过基线的事实也要进入来源漏斗。
每个候选尝试一行 `candidate_attempts.jsonl`，每次实际模型请求一行 `request_attempts.jsonl`。
批请求只计一个物理请求，但列出所有fact与candidate；技术失败、重试、审核、复跑、可选MCQ全部计入。
候选文件允许pending，但只有已审核、实际生成的事实错误且复跑成功才进入主产出分子。

候选行必须有：

```json
{"candidate_id":"f1-a1","base_fact_id":"f1","split_assignment":"development","attempt_number":1,"generator_strategy_version":"v1","manipulation_family":"explicit_false_assertion_v1","terminal_status":"completed","static_review_passed":true,"original_correct":true,"neutral_correct":true,"targeted_correct":false,"targeted_distractor_hit":true,"error_kind":"factual","replication_passed":true,"exclusion_reason":null}
```

请求行示例（真实运行逐次记录，不用此示例冒充完整调用量）：

```json
{"request_id":"req-001","split_assignment":"development","base_fact_ids":["f1"],"candidate_ids":["f1-a1"],"role":"target","terminal_status":"completed","total_tokens":256,"cost_usd":null}
```

`role`为 baseline/generator/reviewer/target/replication/mcq_aux；重试拥有新request_id。
成本缺失填null；token缺失填null并披露未验证预算，不补成0。若provider按批返回usage，只在该批请求计一次。

```bash
python scripts/summarize_open_discovery.py \
  --policy configs/open_factual_discovery_v1.json \
  --candidates <run>/candidate_attempts.jsonl \
  --requests <run>/request_attempts.jsonl \
  --output <run>/discovery_efficiency_summary.json
```

该离线检查器验证已提供账本并拒绝超限、第三轮、成功后再生、holdout反馈、重复请求ID和无请求记账的候选。
它不自动调度模型，也不能单凭账本证明外部provider日志完整；执行方须在每轮发请求前执行同一预算限制并预留复跑开销。

主指标：

- 唯一复现缺陷事实数 / 所有尝试来源事实数；每100次模型请求得到的唯一复现缺陷事实数。
- token使用完整时：每百万token唯一缺陷事实数；定价完整时：每个唯一缺陷事实的实际成本。
- 对照稳定性、静态排除、语义未定、技术失败、复现失败和扰动类型分布。
- 定向/非定向和语言特异结果分别报告，按事实聚合，不用候选数/语言数/变体数扩充独立样本量。

## 5. 当前EN/ZH分析入口

```bash
python scripts/analyze_qwen3_option_free.py --run-dir <run>
```

该分析器读取 `inputs/g2a/option_free_three_arm_inputs.jsonl` 和 `results/open_results.jsonl`，
支持实际fact数量及每事实一个或多个攻击目标；不要求960行或两个variant。
`inputs/mcq_labels.jsonl`可省略，缺失MCQ标签为null，不影响开放缺陷标签。
缺失的候选结果保留为未知事实状态。新版scoring_version另记，不覆盖旧汇总。
这仍是EN/ZH结果接口；六语运行器接入在新执行配置中单独完成，不宣称当前旧160 runner支持任意新cohort。

## 6. 机制准入与独立验证

开放缺陷冻结后，绑定同一exact-HF运行环境、开放输入与评分合同，保留抗扰动对照，再做PNT机制假设检验和修复。
开放行为缺陷本身不证明具体机制；修复需要no-op/random与保持/伤害指标，并在独立事实上确认。
生成器可读取Development的有限行为反馈；静态终审独立执行，不能用“模型翻转了”倒推材料正确。

现有32条Validation已在2026-10-17用于固定修复确认，结果为G3未通过；它不能作为新策略的未见验证集。
Sealed按既定规则保持。新生成策略需要新的未暴露确认范围或明确标注旧集为开发证据，不能因调整流程自动重新开启holdout。

## 7. 任务清单

- [x] 新候选取消MCQ阳性准入与第三foil要求，直接开放三臂判定。
- [x] 固定首轮+最多一次反馈修订的policy，支持离线预算与效率核验。
- [x] EN/ZH分析器支持无MCQ、小cohort和单扰动，缺失状态不丢分母。
- [ ] 在实际选定的目标模型运行器接入新cohort、预算检查与逐请求账本；不复用旧160固定覆盖假设。
- [ ] 固定新Development成员与输入版本后执行有界生成，保存全部失败与对照。
- [ ] 冻结复现缺陷后进行机制与修复，另用独立范围确认。

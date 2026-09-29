# 五语扩展：恢复执行与实验输入口径

更新：2026-09-29。本文是五语扩展的执行与设计补充，不改写已冻结的 160 条协议。
当前用户授权：恢复五语翻译/独立审核、修复失败停止机制、明确后续实验输入与比较口径。
已有 160 条 PNT 闭环暂停；本文不启动 HF、tokenizer、行为筛查、hidden states、向量或干预。

## 1. 固定范围与当前产物

来源链：8,969 个原始 base facts → 7,950 个 accepted-only candidate targets →
7,663 个中文审核通过 targets → 4,513 个关系分组可用 targets。
五个目标语言固定为 `ar / de / es / ja / sw`，英语是内容源；中文只作为来源筛选，
不作为转译中间语言，也不混入五语主比较。

当前投影每事实只有一个已接受的 distractor 和一个 Neutral reference candidate。
它是翻译与后续输入建设的起点，不等同于最终三臂 MCQ，更不等同于 PNT 准入。
2,346 条 `broad_rule_candidate_partition` 与 2,167 条
`existing_probe_relation_candidate_partition` 均不得自动升格为正式 `probe_relation_id`。

原运行：
`data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/full-4513-multilingual-review-r09-v1/`。
它完成了处理但没有产生通过审核的五语记录：4,331 条生成失败，182 条审核失败，
记录的服务错误为 `403 / insufficient_user_quota`。这些是技术失败，不是语义不合格。

## 2. 恢复执行合同

继续采用一次语义生成、一次独立模型审核、零自动内容修复。
`accept/reject` 是已完成的语义裁决；`failed/pending` 是技术状态。
`--retry-failed` 只补技术失败与未完成任务，不重审已完成的 reject，也不重生成已成功的翻译。

v2 runner 的停止规则：

- 额度耗尽、认证/权限错误、模型或路由身份错误立即停止新增请求。
- 连续三个顶层批次全部技术失败时停止；成功批次重置计数。
- 最多保留当前并发波次中已发出的请求，收齐结果后写检查点，不再开始下一波。
- JSON/schema 类问题仍可按原规则拆分批次；额度与权限错误不通过拆批重试。
- 未尝试的记录保留 pending，不生成伪造的 reject/failed；已有失败历史保留。
- 退出码 `3` 表示可恢复的服务停止，`2` 表示处理结束但仍有技术失败，`0` 表示所选范围技术处理完成。
  语义 reject 本身不构成执行失败；是否全量以及是否全部 accept 单独报告。

代码变更会改变 run fingerprint，因此使用新 recovery 目录，不覆盖原运行。
`--resume-from` 核对原 manifest、checkpoint SHA、模型/目标路由、输入、语言与共享 runtime；
仅允许执行逻辑升级。182 条成功生成记录逐字节继承，已完成语义审核也原样继承。
新请求使用新 fingerprint，来源由 `recovery_lineage` 绑定；源运行保持不变。
同一 recovery 目录后续续跑同时传 `--resume --resume-from <原manifest>`。

v3 支持从已退出进程的不可变 `interrupted_checkpoint_snapshot` 继续升级，
校验该快照中的 checkpoint SHA 与祖先 manifest 绑定，继承所有已绑定的历史 fingerprint。
`--generation-workers` 和 `--review-workers` 分别控制生成与审核线程数，
对应 provider semaphore 同步调整，实际生效值写入 `execution_policy`。
在运行目录创建 `STOP_AFTER_WAVE` 可要求收齐并保存当前波次后退出；
状态为 `stopped_after_checkpoint`，恢复前须移除该文件。

2026-09-29 并发测试采用固定静态事实、batch=4、一次请求、无拆批重试，
分别测量 4/8/16 个在途请求。4/8 档使用同样的前 32 条事实，16 档使用前 64 条，
因此吞吐比较是有限规模的容量测试，不能当作翻译质量的统计结论。
旧主任务在测试期间退出并保留检查点，不与测试流量叠加；测试输出不进入正式翻译池。
报告区分服务错误、JSON 解析失败、schema 校验失败与有效事实/分钟。
该次测试只覆盖生成端，当时审核并发保持 2；任何 PNT/HF 执行继续暂停。

正式续跑前用一个既定样本完成生成与审核的连通性检查。
检查只确认链路能工作，不能证明整批额度足够；运行中仍执行上述停止规则。
新运行的 `redacted_events.jsonl` 只统计本次恢复后的请求，旧请求与曝光统计通过
`recovery_lineage.prior_exposure` 保留，两者不能把 unique fact 数直接相加。

2026-09-29 恢复记录：真实单条预检五语 5/5 接受，生成/审核技术失败均为 0。
第一次全量恢复目录为
`data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/full-4513-multilingual-review-r09-recovery-v2-20260929/`。
各运行目录的 `run_manifest.json` 记录该版本状态，`execution.log` 记录进度，
`launcher.json` 保存 PID 与完整无凭据命令；不能把“已启动”写成全量完成。
预检目录是相邻的 `full-4513-multilingual-recovery-preflight-20260929-v2/`，
不并入正式恢复运行的样本或模型输出。

如进程已停止且外部服务已恢复，可读取 `launcher.json` 的 `argv` 并追加 `--resume` 续跑；
保留其中的 `--resume-from`。若状态为 `running`，先核实原进程是否仍在运行；
同目录 writer lock 会拒绝第二个并发写入者。

### 2026-09-29 高并发实测与当前运行

| 并发 | 请求数 | 格式合格请求 | 服务失败 | 有效事实/分钟 | 延迟中位数 / P95 |
|---|---:|---:|---:|---:|---|
| 4 | 8 | 4 | 1 次 300 秒超时 | 3.197 | 67.7 / 300.3 秒 |
| 8 | 8 | 2 | 0 | 6.784 | 66.6 / 70.7 秒 |
| 16 | 16 | 4 | 0 | 9.024 | 64.8 / 106.4 秒 |

4 并发出现超时后，额外以 2 并发复核两个批次：均在 67.3 秒内返回，无服务错误，
但两批仍不满足格式校验。随后才进行 8/16 档测试。全部测试未观察到 429。
表中合格指 JSON/schema 合格，不是独立语义审核接受；多数格式失败是批内事实 ID 不匹配，
也有语言 key 或字段问题。不同阶段重复请求的输出有波动，不能把这些小样本比例当成并发导致的质量效应。
16 档有效吞吐比 8 档高约 33%，但 P95 延迟更长；结论限于本次短测。

生成并发升级时采用 **翻译 16 并发、审核 2 并发、batch=4**，当时目录为
`data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/full-4513-multilingual-review-r09-recovery-v3-c16-20260929/`。
该次恢复继承 382 条已成功生成记录和所有失败历史。
旧 v2 目录的 `checkpoint_snapshot_for_concurrency_upgrade.json` 绑定该次继承的完整检查点，
旧正式记录按来源继承，探测输出独立保存，不作为正式翻译结果。

按用户随后指示，当前主任务已切换为 **翻译 16 并发、审核 8 并发、batch=4**。
两个阶段依次执行：生成阶段上限 16，审核阶段上限 8；线程数与 provider semaphore 同步设置。
当前目录为
`data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/full-4513-multilingual-review-r09-recovery-v3-c16-r8-20260929/`。
旧 c16 运行经 `STOP_AFTER_WAVE` 收齐在途请求、保存检查点后正常停止，状态为
`stopped_after_checkpoint`，writer lock 已释放；同目录的
`checkpoint_snapshot_for_review_c8.json` 绑定完整检查点与停止后的 manifest。
新运行逐字节继承 668 条成功生成记录、其余技术失败记录和审核检查点，已核验
`generation_workers=16`、`review_workers=8`、`aliyun=16`、`openai=8`。
独立审核仍待生成阶段完成后执行；审核 8 并发尚无吞吐实测，不能据此承诺四倍加速。
若需重启，使用当前目录的 `launcher.json`，保留 `--resume-from` 并追加 `--resume`。

测试明细位于同级目录：`multilingual-concurrency-probe-20260929-v1/`、
`multilingual-concurrency-timeout-recheck-20260929-v1/`、`multilingual-concurrency-probe-high-20260929-v1/`。
新运行仍保留失败停止、检查点及拆批恢复；PNT/HF 继续暂停。

## 3. 扩展版最终实验输入

后续采用 **一个指定攻击目标、三选一、三臂**，不复制旧 160 的双攻击 variant：

| 对象 | 扩展版定义 |
|---|---|
| `gold` | 同一事实的正确答案与各语 aliases |
| `designated_distractor` | 继承当前投影已接受的那个 distractor，不按目标模型结果替换 |
| `non_target_foil` | 另一个已验证错误且不与前两者 alias 重叠的答案，只作第三选项 |
| Original | 无额外上下文的三选一 |
| Neutral | 含已审核匹配 Neutral 上下文的同一个三选一 |
| Targeted | 含指向 designated distractor 的真实干扰上下文的同一个三选一 |
| Natural | 不含选项、不含扰动上下文的自然完成输入 |

`non_target_foil` 和最终 Neutral/Targeted 上下文是明确待做项，不由当前翻译结果冒充完成。
只对后续预先选定的行为 cohort 补齐，不要求先补齐全部 4,513 条；不回退到二选一。
优先采用已审核 source wrong option，再采用已验证同 answer type 的候选；固定排序、来源与排除原因，
禁止基于 HF 成败选 foil。补齐的 foil、上下文与最终渲染输入需独立语义审核。

各语言使用同一个 base fact、同一组答案身份、同一攻击目标和同一选项排列。
选项排列在目标行为运行前按事实哈希固定；报告 gold slot 分布，不能按语言重新随机。
Neutral 与 Targeted 在各语言内匹配主题、长度与风格；审核真实陈述、关系保持、唯一答案和翻译等价。
翻译审核接受不自动替代新增上下文及最终组合输入的审核。

EN + 五语共有六种语言。每事实每模型为 `6 × 3 = 18` 个 MCQ 输入，
另有 `6` 个 Natural 输入；这是未来完整输入的数量合同，不是已生成或已执行的数量。
五组 EN–目标语比较共用同一份 EN 输出，不为每个目标语重复调用或重复计样本。
若以后报告中文参照，另列 EN–ZH 结果，不改变五语主分析分母。

## 4. 比较口径与统计单位

以 `base_fact_id` 为独立事实计数；语言、arm、alias、模型和重复调用都不增加独立样本量。
保留父级 leakage component 与 split；五语过滤不重新划分数据。
当前 4,513 条的继承分布为 Development=2,710、Validation=901、Sealed=902。
这些是候选范围，正式运行仍需冻结所选 cohort 的成员清单和输入 SHA。

| 集合 | 定义与用途 |
|---|---|
| 全翻译范围 U | 固定 4,513 条，所有失败与排除都留在漏斗中 |
| 单语可用集 A_lang | 该语言翻译审核通过的事实；报告语言覆盖率及补充行为结果 |
| 五语共同集 C | 五个 A_lang 的交集，用于同事实跨语言主比较 |
| 最终行为集 B | C 中按行为盲态规则选定且六语最终输入均合格的事实；冻结后才执行 |
| PNT 子集 | 在 B 内另满足正式 relation、exact-HF 与机制合同的事实，当前不执行 |

C 不足时不临时改主集合以追求显著结果；可以另报 EN–单语配对子集，但标作不同 estimand，
不能直接比较不同样本上的准确率并称为语言效应。
保留单语合格记录，即使另一语言失败也不全部丢弃。C/B 都报告 relation/source/split 分布，
方便识别共同集筛选造成的构成变化。

每种语言至少报告 `U → generation completed → review completed → accept/reject/technical failed`
以及 `C → 最终输入合格 → 行为终局 → directed/off-target/neutral-unstable`。
缺失不是答错；提供完成率与可评估分母，不能静默删失败。

行为主定义：Original=gold、Neutral=gold、Targeted=designated distractor。
EN 三臂均正确只定义更窄的目标语特异子集，不否定其他 directed 事件。
同时报告全体冻结事实的 Targeted−Neutral 正确率变化及 target-hit 变化，
以及同事实 `(Targeted−Neutral)_lang − (Targeted−Neutral)_EN`。
这些是配对行为对比，不单凭差异宣称机制因果证据。

比例给出 `n/N` 与 95% CI；成对差异在事实层聚合，重抽样时以 leakage component 为块，
保留同事实各语言/arm 的配对。关系小格只作描述；若对五语同时作显著性结论，用 Holm 校正。
自然 EN 正确的限定只用于自然跨语言知识缺口分层，不作为一般行为结果的事后删除条件。
不直接比较跨语言的原始 token rank 或 sequence log probability 大小；优先比较各语言内配对变化。

## 5. 冻结与后续边界

当前中文筛选与关系筛选决定了研究适用范围，报告不能外推成原始 8,969 条的无偏总体估计。
不为消除这一限制重建全部数据，也不追加全量人工复审。

正式行为前仍需：选定 B 的成员与最终输入、补齐 foil/context 审核、完成相关历史曝光合同、
输出绑定同一版本的 review/split freeze，之后才解析目标 HF 身份与 tokenizer。
只在实际发现新增泄漏关系时处理 component；不因增加语言机械重做所有语义闭包。
父级支持 donor 可以不属于最终 targets，但必须在已绑定的父 support cohort 内并符合既有 split 约束；
不能把 support-only donor 自动变成行为目标、vector source 或独立样本。

翻译辅助模型的静态文本接触单独记入曝光记录，与目标模型行为曝光区分。
本次不打开 Validation/Sealed 行为，不运行任何旧 160 或新扩展 PNT。
恢复 PNT 须等用户后续指示；不因五语翻译完成自动触发。

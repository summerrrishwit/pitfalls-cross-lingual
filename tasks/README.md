# `tasks/` 状态与执行入口

## 2026-09-30 当前执行范围

当前执行五语翻译/独立审核恢复与失败停止机制，扩展输入及比较口径见
[`task_multilingual_expansion_protocol.md`](task_multilingual_expansion_protocol.md)。
**按用户指示，已有 160 条 PNT 闭环暂不执行。** 下文保留其历史状态与协议，
其中“当前下一步”不构成本次恢复 HF、向量或 intervention 的执行指令。
按用户最新指示，当前活动运行是 `full-4513-multilingual-review-r09-recovery-v3-c16-r8-20260929`：
翻译 16 并发、审核 8 并发、每批 4 条，两个阶段依次执行；线程数与接口上限同步设置。
旧 c16 运行已保存检查点并停止，新运行通过绑定快照继承 668 条成功翻译及失败历史。
2026-09-30 09:06（北京时间）已在同一运行目录断点续跑：启动前累计成功翻译
3,279/4,513 条，剩余 1,234 条；完成生成后自动进入独立审核。恢复前单条真实预检
五语全部通过、翻译和审核技术失败均为 0；该预检独立保存，不并入正式样本。
本次启动记录为 `launcher_resume_20260930T010619Z.json`，日志为
`execution_resume_20260930T010619Z.log`；实时状态以该目录 `run_manifest.json` 为准。

## 唯一权威的 160 条流程

当前 160 条数据的执行设计以
[`task_160_mcq_pnt_protocol.md`](task_160_mcq_pnt_protocol.md) 为唯一权威入口。

中心思想是：

```text
全 160 静态内容生成 + Codex 审核
                ↓ G0A_STATIC_CONTENT_POOL_FROZEN
Development 96：四模型代理筛查
                ↓ P0_DEV_PROXY_SCREEN / P0_PROXY_REPORT
exact-HF 身份、tokenizer 与最终渲染冻结
                ↓ G0B_EXACT_HF_RENDER_FROZEN
exact-HF 完整重跑 Development 96
                ↓
PNT 归因、干预与修复
                ↓
Validation / Sealed 独立确认
```

`Development / Validation / Sealed` 的主要目的不是证明扰动生成器能泛化到所有关系，
而是防止用同一批案例发现缺陷、选择机制、调整干预并再次证明修复成功。
代理模型只用于挖掘和比较行为，不能替代 exact-HF 的重跑或 PNT gate。

## 当前客观状态

- 数据规模：160 个 `base_fact_id`，四个候选关系各 40 条。
- 现有划分：`Development=96`、`Validation=32`、`Sealed=32`。
- `staging-v3` 已完成 160 个 facts、320 个 perturbation groups 的静态上游生成，Simulation=0。
- `review-v5` 已完成 160/160 条 Codex 终局裁决：`156 accept + 4 revise`，全部 `human_gold=false`；
  decision SHA-256 为 `beab077c3aed038242ae840db3aa7616b3832ec46d79c8d87316b9fa92969457`。
- 两条行为盲态 Neutral 修复已通过 `context-repairs-v1` overlay 纳入复审；overlay SHA-256 为
  `6ae1351f39ca251ca50e3b6fdc0ae1674cf8e8562d1052f4d16807bdeca2aa2b`。
- 经 `review-v5` 裁决后，`resolved-v1` 已物化 160 条 resolved rows、320 个双语
  Neutral/Targeted variants（每事实 2 个）和 1,600 个唯一静态输入（每事实 10 个）；bundle SHA-256 为
  `601f23902b309cca12df3da07c6364214b408f8c91ac53808363bcafb138f224`。
- 当前唯一权威链是 `staging-v3 → review-v5 → resolved-v1`；不得回退到 `staging-v2`、
  仅含单项修复的 `review-v4`，也不得复用旧 `/private/tmp` closure v1。
- semantic closure 已在 `semantic-closure-v2-final/semantic_closure_manifest.json` 中按冻结的 bounded
  protocol 完成：358 条 full-pool lexical candidates 与 15 条 current-cohort semantic candidates
  均已终局裁决，去重后为 372 对，重叠决策一致，未发现跨现有 split 的 positive component；
  manifest SHA-256 为 `bbe39cc41ac98cf5d80098c411f38bd8d6afda4aae741df5637639808dcc6957`。
  该结论不声明无界语义召回，仍保持 `semantic_near_duplicate_recall_guaranteed=false`。
- 最终仓库扫描 `historical-exposure-pending-v3` 覆盖 977 个 current-worktree files、455 个 reachable
  Git blobs 和 196 个含 holdout identifier 的 artifacts，记录级行为曝光命中为 0。scope-owner
  attestation 已绑定，64/64 holdout checks complete、registry=0；complete manifest SHA-256 为
  `451a6290eeadfdab8dabca0f86e89cb416972a8fb3236587216787dad1db67ff`。
- `G0A` 已正式冻结：`frozen-v1/static_freeze_manifest.json` 的状态为
  `g0a_static_stimulus_frozen`，SHA-256 为
  `f033c9cf2d4529b2cde03878a4d01b84a462d167ce8aaaaec13cb3e9972e8636`。
- P0 后、G0B 前的补充冻结已保存到 `pre-exact-hf-protocol-v1/`：
  `development_fold_manifest.json` SHA-256 为
  `463dfac6cb786f2baf756af615e758e04d7257f7ce639d501aa0255780958c8e`，
  `perturbation_protocol_manifest.json` SHA-256 为
  `b43b927166c469feb786a66fbed277f687e775bad1ced0b9ce3e2032c513cbac`。
  该补充冻结不构成 P0 前预注册，也不改写 G0A 或 P0；它对随后执行的 exact-HF replay/PNT
  前瞻有效。manifest 中的 `exact_hf_resolution_status=pending_g0b` 是生成时状态，现已由下述 G0B/G1
  产物取代。
- 活动配置 `configs/factual_perturbation_zh_dual_distractor_proxy_v10.json` 已生成 3,840 个
  Development-only logical calls（四模型各 960）；run ID 为
  `static-g0a-dev96-four-model-v10-20260916`。四模型 representative probe 已全部通过：
  `completed_model_count=4`、`failed_models=[]`、`full_run_authorized=true`。
- v10 将 Gemini 固定为 `max_output_tokens=512`；主 `run`、audit、three-arm analysis 与 report
  已完成：3,840/3,840，四模型各 960，Development=3,840、Validation=0、Sealed=0，
  `proxy_screen_complete=true`、`operational_reasons=[]`。
- `gate_passed=false` 的唯一原因是 `proxy_behavior_only_not_pnt_eligible`。这不是代理运行失败；
  它表示结果仅为 proxy behavior，`pnt_eligible=false`、`pnt_authorized=false`，不能称为 HF/PNT 证据。
- 旧 run `static-g0a-dev96-four-model-v9-20260916` 在 734/3,840 处停止（730 completed、
  4 Gemini failed）；失败记录是在 `max_output_tokens=256` 下产生的 `truncated_json`。
  v9 已标记 `superseded_do_not_resume` 并保留为审计历史，不得恢复或与 v10 结果混合。
- 旧 run `static-g0a-dev96-five-model-v8-20260916` 因 `bailian/deepseek-v3.2` 路由失败而
  fail-closed，现标记为 `superseded_do_not_resume` 并保留为审计历史；不得覆盖、续跑或与 v9/v10 混合。
  `bailian/deepseek-v3.2` 不再属于活动 roster，也不以其他 DeepSeek 版本替换。
- exact-HF 已固定为 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`。G0B v3、
  Development 960 条行为重跑及独立重复均已完成；v3 行为结果为 96 个 directed variants/57 facts、
  14 个 ZH-specific variants/11 facts、22 个 raw resistant variants/15 facts。排除 final sequence-score
  精确并列后，稳健 resistant 为 21 variants，matched directed 为 44，unmatched directed 为 52。
- 自然 EN/ZH baseline v4 已完成 192/192，并由 Codex 语义复核为 `codex_proxy`、`human_gold=false`：
  64 accept、125 reject、3 defer；语义正确为 EN 36/96、ZH 28/96，natural PNT gap=10、
  induced-only gap=14、natural EN failure=58。
- Development MCQ decision-position attribution v2 已完成 960/960：保存 36 层 Logit Lens 与
  `[960,36,4096]` BF16 `resid_pre`。正式运行与独立复跑的 metrics/index/activation SHA 全部一致，
  行为 final-token rank 与 choice order 差异均为 0；稳定性审计 SHA-256 为
  `253e69b10d8260e9eb36fc54aebc06f314f7d64fd46e42afbf07b241fdee92f1`。
- 旧 attribution v1 使用单前缀前向，出现 62 个全词表 rank 差异和 29 个 choice-order 差异，已标记为
  非权威诊断产物；不得与 v2 混合。
- Development-only 的 G2A 文本合同已物化到 `g2a-option-free-preparation-v1/`：960 条不含选项的
  Original/Neutral/Targeted 输入，以及 1,920 条四折 OOF/final vector-source prompt spec。离线审计已核对
  96/32/32 隔离、四折 source/evaluation 无重叠、5-shot donor 同关系且不同 component、Original 与自然
  baseline 用户内容一致、Neutral/Targeted 只通过 context 改写，以及三个答案组 alias 不相交；contract
  manifest SHA-256 为 `6e9ee3317bc8b040e3eefb3ad784d305264cec6b2506834639367e4a2cf7af6f`，
  audit SHA-256 为 `73129a5172a3b7a5975dd7a4515558ef2de7115cf910eaaebb5355047318b3d7`。
- 该 G2A 包仍是 `offline_contract_prepared_pending_exact_tokenizer_audit`，没有新增模型输出、tokenization、
  hidden states、vectors 或 intervention；Validation/Sealed 使用量仍为 0。自然语义审核的 3 个 `defer`
  已按缺失值处理。另已行为盲态冻结 13 个双语 unrelated facts（26 个输入），只作 retention regression，
  不进入 cohort/fold/vector source；其小样本 2% 阈值等价于不允许出现任何新增受损事实。Development
  vector intervention 仍须等 exact-tokenizer 审计后才可授权。control manifest SHA-256 为
  `c06d92a946b697560aa9dbffb174b14d0f0b734817b31d117ad89a70d4716768`。

逐模型三臂 proxy 标签如下；variant 不是独立样本量，且没有生成 panel/union 标签：

| 模型 | completed calls | directed variants | ZH-specific strict variants |
|---|---:|---:|---:|
| `qwen3.6-27b` | 960 | 55 | 28 |
| `gemini-3.7-flash` | 960 | 168 | 102 |
| `gemma3:12b` | 960 | 91 | 44 |
| `llama3.1:8b` | 960 | 41 | 14 |

`panel_union_reported=false`；上述值必须继续逐模型报告，不能合并成共享正式标签。

本次 v10 P0 的正式输入由下列 v3 semantic-aware bundle 经 review/freeze 产生；后续新版本也不得绕过该 lineage：

```text
data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/preperturbation_v1/postclosure_preperturbation_v3/postclosure_preperturbation_behavior_input_bundle.jsonl
```

不能把 PATH adapter 的 `review_only_base_facts.jsonl` 直接当作实验输入。

## 权威执行顺序

```text
Stage 0A 全 160：事实 / 翻译 / alias / 双 distractor / 两对上下文 / 泄漏闭包 / split 静态冻结
    ↓ G0A_STATIC_CONTENT_POOL_FROZEN
P0       Development 96：四代理模型双 variant 行为筛查（每模型 960，共 3,840 logical calls）
    ↓ P0_PROXY_REPORT（不解锁 PNT）
Stage 0B exact-HF：模型身份 / tokenizer / template / scorer / 1,600 个最终 render 冻结
    ↓ G0B_EXACT_HF_RENDER_FROZEN
Stage 1  exact-HF 完整重跑 Development 96（960 个输入，不按代理结果过滤）
    ↓ G1_BEHAVIOR_PROTOCOL_FROZEN
Stage 2  Development 96：PNT 自然基线、机制分析与干预开发（交叉拟合）
    ↓ G2_REPAIR_RULE_FROZEN
Stage 3  Validation 32：固定协议的一次确认，不再调参
    ↓ G3_VALIDATION_CONFIRMED
Stage 4  Sealed 32：最终一次性修复、英文保持与无关事实回归
    ↓ G4_FINAL_REPORT
Stage 5  是否扩到 2,450：另行决策，不属于当前 160 闭环
```

代理行为只允许使用 `Development=96`。Validation 和 Sealed 在相应 gate 前保持行为不可见，
但可以在不读取任何行为输出的前提下完成静态生成和 Codex 审核。

## 双 distractor 与代理标签

每条事实使用同一个三选一 MCQ：`gold + distractor_1 + distractor_2`。两个 distractor 各自有
`Neutral_d` 与 `Targeted_d`，EN/ZH Original 在两个 variant 间共享并只执行一次：

```text
EN/ZH Original                        = 2
D1 的 EN/ZH Neutral/Targeted          = 4
D2 的 EN/ZH Neutral/Targeted          = 4
每 fact、每 model                     = 10 inputs
```

活动 v10 的四个 Development 代理模型固定为：

```text
qwen3.6-27b
gemini-3.7-flash
gemma3:12b
llama3.1:8b
```

其中 `gemini-3.7-flash` 的 v10 输出预算固定为 `max_output_tokens=512`。

每个模型、每个 distractor 独立产生两级嵌套标签：

```text
proxy_directed_candidate@model_id:
  ZH Original = gold
  ZH Neutral_d = gold
  ZH Targeted_d = designated distractor

proxy_zh_specific_strict@model_id:
  proxy_directed_candidate
  + EN Original/Neutral_d/Targeted_d = gold
```

“strict 放宽”落实为上述两级 gate：`proxy_directed_candidate` 是高召回主 gate，
只要求中文 Original/Neutral 正确且 Targeted 命中指定 distractor；英文三臂只用于更窄的
`proxy_zh_specific_strict`，不否决定向缺陷。不得进一步删除 Neutral、接受 off-target 错误，
或用四模型投票替代逐模型判定。

不得投票或把四模型 union 当作正式标签。活动 exact-HF 已重跑全部 Development 96，
包括所有代理模型均未翻转的事实；只有 exact-HF 自己重新确认的标签可以进入 PNT。

Codex 可替代人工做静态终局裁决。当前 `static-g0a-codex-decision-v1` 的实际字段为：

```text
reviewer_type = codex_proxy
decision = accept | revise | reject | defer
terminal_status = completed  # 仅 terminal 记录
human_gold = false
```

若导出通用审核字段，可以显式映射成 `adjudicator_type=codex` / `review_status=codex_adjudicated`，
但不得声称当前 JSONL 原生含有这些字段。Codex 审核不得查看代理/目标模型行为结果，也不得写成
`human_gold` 或独立人工验证。

## 旧任务文档状态

下列文件保留为历史设计记录，不能单独作为执行说明：

| 文件 | 状态 | 原因 |
|---|---|---|
| `task_1_160_data_pilot_perturbation.md` | `SUPERSEDED` | 随机抽取 50 条会消费 Validation/Sealed，且把 MCQ flip 直接称为 PNT 候选 |
| `task_2_160_data_full_perturbation.md` | `SUPERSEDED` | 混跑全部 160、混合模型身份，并把 MCQ 结果直接导出为 PNT 数据 |
| `task_3_data_quality_audit.md` | `REFERENCE_ONLY` | 抽样质量审计不能代替逐项 formal review/freeze |
| `task_4_distractor_quality_enhancement.md` | `HISTORICAL_REFERENCE` | 旧校准草案；正文命令/阈值不能用于当前链，任何新校准都必须发布新 G0A 版本并完整重跑 P0 |
| `task_5_2450_data_preprocessing.md` | `DEFERRED` | 需等 160 条 attribution→repair 闭环通过后另行授权 |
| `task_6_neutral_stability_optimization.md` | `HISTORICAL_REFERENCE` | 旧 73 条诊断的优化草案；不得用于当前 160 事后改 context、删 Neutral 或挑模型保住候选数 |

这些旧文档中的预算、耗时、候选率、候选数量门槛和命令均是未验证设想，
其中的旧 standalone 脚本名和示例参数不得复制执行。当前统一 runtime 已实现以下代理链：

```text
prepare-static-proxy-run
→ probe-static-proxy-run
→ run
→ audit
→ analyze-three-arm
→ report
```

这些 subcommand 的存在只表示执行入口已实现；`prepare-static-proxy-run` 仍必须读取真正通过的
`g0a_static_stimulus_frozen` manifest，代表性四模型 probe 全部通过后才允许 `run`。当前 v10 已满足
该前置条件，并已完成 `run → audit → analyze-three-arm → report` 与 `P0_PROXY_REPORT`；该完成态
不等于 exact-HF/PNT gate 通过。

## 五条不可变边界

1. 代理输出只叫 `proxy_directed_candidate@model_id` 或 `proxy_zh_specific_strict@model_id`，不是 HF/PNT 缺陷证据。
2. 两个 distractor 都进入行为测量，但独立统计单位始终是 `base_fact_id`；variant、模型、语言、arm、层和 scale 不能膨胀样本量。
3. 四代理只跑 Development；Validation/Sealed 不得产生代理行为输出。
4. exact-HF 已完整重跑 Development 96；后续任何新版本仍不得只复测代理阳性。
5. 缺陷、hidden-state、vector、intervention 和 repair 必须绑定同一个 exact-HF 模型身份。

## 当前下一步

G0B、G1、自然 baseline、Development MCQ decision attribution v2 与 G2A 文本级离线合同均已完成；
旧行为 v8/v9 和 attribution v1 继续只作 superseded 审计证据，不得恢复或混合。HF 暂停期间不启动任何
新 hidden-state 或 intervention 运行。恢复同一冻结 checkpoint/tokenizer 后，先对 G2A prompt 做 exact
tokenizer render、长度和 answer-prefix 审计；behavior-blind unrelated-fact control 已完成文本冻结，届时
一并建立 no-op baseline。exact-tokenizer 审计通过前，Development intervention 继续 fail-closed，
Validation/Sealed 继续保持未开启。

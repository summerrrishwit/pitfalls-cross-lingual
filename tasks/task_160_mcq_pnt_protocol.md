# 160 条 MCQ 扰动 → PNT 归因与修复协议

**状态：权威设计；G0A/P0/G0B/G1、Development 自然 baseline 与 MCQ decision-position attribution v2 已完成；answer-text PNT vector/intervention、Validation、Sealed 仍未授权**  
**范围：当前 160 条关系平衡 cohort**  
**替代：旧的“随机 50 pilot → 全 160 扰动 → 2,450 扩量”顺序**

## 1. 研究目标与声明边界

本协议回答：

> 能否用定向 MCQ 扰动，在一个 exact-HF 模型上稳定挖出方向明确的中文弱点，
> 再用 PNT 解释并修复该弱点，同时保持 Original、Neutral、英文和无关事实性能？

本协议不试图证明：

- 扰动生成方法能泛化到所有关系或全部数据；
- 不同模型会产生同一个弱点；
- MCQ flip 自动等价于自然开放补全中的事实召回错误；
- API/Ollama 的行为信号可以解释另一个 HF checkpoint 的内部机制。

核心分工是：

```text
MCQ = 弱点放大器和缺陷挖掘器
PNT = 机制分析与修复方法
Split = 防止 PNT 归因和修复自我验证
```

当前执行边界固定为：

```text
G0A_STATIC_CONTENT_POOL_FROZEN
    ↓
P0_DEV_PROXY_SCREEN（只跑 Development 96）
    ↓
G0B_EXACT_HF_RENDER_FROZEN
    ↓
exact-HF 完整重跑 Development 96
    ↓
PNT 归因与修复
```

`P0_DEV_PROXY_SCREEN` 只产生四个代理模型各自的行为标签，既不替代 `G0B`，也不解锁 PNT。

所以 `Validation / Sealed` 不以“扰动候选率必须复现”为主要目标；它们用于检验：

1. 冻结规则是否还能识别相同类型的缺陷表型；
2. Development 中选择的机制解释和干预是否在独立事实上成立；
3. 修复是否没有损伤英文、Original/Neutral 和无关事实。

## 2. 当前起点与执行状态

当前已有 160 个 `base_fact_id`，四个候选关系各 40 条，现有顶层划分为：

| Split | 每关系 | 总数 | 角色 |
|---|---:|---:|---|
| Development | 24 | 96 | 缺陷挖掘与 PNT 方法开发 |
| Validation | 8 | 32 | 冻结机制与修复规则的独立确认 |
| Sealed | 8 | 32 | 最终一次性评估 |

截至 2026-09-17，静态阶段已冻结，活动四模型 v10 已完成 `P0_DEV_PROXY_SCREEN` 与
`P0_PROXY_REPORT`：

- `staging-v3` 已完成 160 个 facts、320 个 perturbation groups 的静态上游生成，Simulation=0；
- `review-v5` 已完成 160/160 条 Codex 终局裁决：`156 accept + 4 revise`，全部 `human_gold=false`；
- 两条行为盲态 Neutral 修复保存在 `context-repairs-v1/context_repairs.jsonl`，SHA-256 为
  `6ae1351f39ca251ca50e3b6fdc0ae1674cf8e8562d1052f4d16807bdeca2aa2b`；
- 经 `review-v5` 裁决后，`resolved-v1` 已物化 160 条 resolved rows、320 个双语
  Neutral/Targeted variants（每事实 2 个）和 1,600 个唯一静态输入（每事实 10 个），SHA-256 为
  `601f23902b309cca12df3da07c6364214b408f8c91ac53808363bcafb138f224`；
- 描述性 context-length audit 已完成，但不设置事后长度硬阈值；
- 活动 run `static-g0a-dev96-four-model-v10-20260916` 已完成并审计 3,840/3,840 个 Development
  logical calls，四模型各 960；Development=3,840、Validation=0、Sealed=0。四模型 representative
  probe 全部通过，`completed_model_count=4`、`failed_models=[]`、`full_run_authorized=true`；
  Gemini 的 `max_output_tokens=512`。
- 完成态为 `proxy_screen_complete=true`、`operational_reasons=[]`；`gate_passed=false` 的唯一原因是
  `proxy_behavior_only_not_pnt_eligible`。这表示代理筛查执行完整，但证据层级不足以进入 PNT，
  不是运行失败，也不是 exact-HF/HF 机制证据。
- 旧 run `static-g0a-dev96-four-model-v9-20260916` 在 734/3,840 个 logical calls 处停止：
  730 条 completed、4 条 Gemini failed；后四条在 `max_output_tokens=256` 下产生 `truncated_json`。
  该 run 已标记 `superseded_do_not_resume` 并保留为审计历史，不得恢复或与 v10 结果混合。
- 旧 run `static-g0a-dev96-five-model-v8-20260916` 因 `bailian/deepseek-v3.2` 连续两次返回
  `503 model_not_found` 而 fail-closed；现已标记 `superseded_do_not_resume` 并保留为审计历史，
  不得覆盖或续跑。`bailian/deepseek-v3.2` 不再属于活动 roster，也不以其他 DeepSeek 版本替换。
- exact-HF `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` 的 Development 行为、
  自然 EN/ZH baseline 与 MCQ decision-position hidden-state/Logit-Lens 已运行；answer-text recall/
  translation vector、intervention 和 repair 尚未运行。

旧的 `freeze-preflight-v1/g0a_finalize_blockers.json` 生成于 semantic final 之前，记录了两个
fail-closed blocker：

```text
semantic_paraphrase_closure_evidence_missing
historical_exposure_evidence_missing
```

semantic blocker 现已在 `semantic-closure-v2-final/semantic_closure_manifest.json` 中按冻结的
bounded protocol 关闭，manifest SHA-256 为
`bbe39cc41ac98cf5d80098c411f38bd8d6afda4aae741df5637639808dcc6957`：358 条
full-pool lexical candidates 与 15 条 current-cohort semantic candidates 均已终局裁决，
去重后共 372 对，重叠决策一致，未发现跨现有 split 的 positive component；replacement 对其余
159 条的显式筛查也已完成。该证据只支持有界定义下的
`semantic_paraphrase_closure_complete=true`，不声明无界语义召回；
`semantic_near_duplicate_recall_guaranteed=false` 仍保留。

已使用该 semantic manifest 重新执行过 preflight；当时
`freeze-preflight-v2/g0a_finalize_blockers.json` 只剩
`historical_exposure_evidence_missing`。该 blocker 随后已由下述 complete evidence 关闭；v1/v2
preflight 现在都是历史快照，当前权威状态只看 `frozen-v1/static_freeze_manifest.json`。

最终仓库扫描已物化为 `historical-exposure-pending-v3`：覆盖 977 个 current-worktree regular
files、455 个 reachable Git blobs 和 196 个含 holdout identifier 的 artifacts，记录级行为曝光命中为 0。
scope owner 的外部目录、另行脚本与外部平台确认已绑定，
`historical-exposure-complete-v1/historical_exposure_manifest.json` 为 `status=complete`，64/64
Validation/Sealed checks 已完成，registry 为 0；manifest SHA-256 为
`451a6290eeadfdab8dabca0f86e89cb416972a8fb3236587216787dad1db67ff`。

`frozen-v1/static_freeze_manifest.json` 已实际生成，状态为 `g0a_static_stimulus_frozen`，SHA-256 为
`f033c9cf2d4529b2cde03878a4d01b84a462d167ce8aaaaec13cb3e9972e8636`。随后
活动配置 `configs/factual_perturbation_zh_dual_distractor_proxy_v10.json` 已通过
`prepare-static-proxy-run`，只把 Development 96 展开为 3,840 个 logical calls；Validation/Sealed
均为 0。活动 run `static-g0a-dev96-four-model-v10-20260916` 的四模型 representative probe 已全部通过，
`full_run_authorized=true`；Gemini 的输出预算固定为 `max_output_tokens=512`。3,840/3,840 个 logical
calls 已按预注册 missing/retry 策略 terminal，`audit`、`analyze-three-arm` 与 `report` 均已完成；
四模型各 960，Development=3,840、Validation/Sealed=0，`proxy_screen_complete=true`、
`operational_reasons=[]`。`gate_passed=false` 仅由 `proxy_behavior_only_not_pnt_eligible` 导致，
`pnt_eligible=false` 与 `pnt_authorized=false` 保持不变。

旧的四模型行为 run `static-g0a-dev96-four-model-v9-20260916` 保留为 superseded 历史证据：它在
734/3,840 处因 4 条 Gemini 256-token 输出预算截断而停止，状态为 `superseded_do_not_resume`。
v10 是新 config/new run，不是对 v9 的原地续跑；两者结果不得混合。

旧的五模型行为 run `static-g0a-dev96-five-model-v8-20260916` 保留为 superseded 历史证据：其
`bailian/deepseek-v3.2` route 失败事实不得删除，但该模型已从活动 roster 移除，旧 run 不得恢复、
覆盖或与 v9/v10 结果混合。这里的 behavioral v8 disposition 不改变
`static-g0a-160-dual-v8-20260916/frozen-v1` 作为当前静态冻结链的权威地位。

当前唯一静态权威链为：

```text
static-g0a-160-dual-v8-20260916/staging-v3
  → static-g0a-160-dual-v8-20260916/review-v5
  → static-g0a-160-dual-v8-20260916/resolved-v1
  → semantic-closure-v2-final + historical-exposure-complete-v1
  → static-g0a-160-dual-v8-20260916/frozen-v1
```

其中 `review-v5/codex_adjudication_decisions.jsonl` 的 SHA-256 为
`beab077c3aed038242ae840db3aa7616b3832ec46d79c8d87316b9fa92969457`。
不得回退到 `staging-v2`、仅含单项修复的 `review-v4`，也不得把旧 `/private/tmp`
closure 结果包装成当前证据。

正式上游只能使用：

```text
data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/preperturbation_v1/postclosure_preperturbation_v3/postclosure_preperturbation_behavior_input_bundle.jsonl
```

当前记录的 bundle SHA-256 是：

```text
4fc5513266b9d8ce2e4a10f7adce0806bc5c35a358cd8f35070dd95943b50c54
```

任何内容变化都必须产生新 manifest 和新 SHA，不能继续沿用该值。
`review_only_base_facts.jsonl` 只是 PATH adapter 的审核视图，不能直接输入正式扰动。

## 3. 不可变的数据与模型合同

### 3.0 证据层级与模型角色

不同角色的输出不能互相冒充。当前静态链固定为：

| 层 | 当前执行者 | 允许做什么 | 不能证明什么 |
|---|---|---|---|
| 来源与确定性规则 | 程序化 checker | 校验 ID/SHA/lineage、split/component、alias-disjoint、候选覆盖 | 不能凭这些规则证明世界事实为真 |
| 来源事实裁决 | 行为盲态 Codex | 逐条裁决 canonical、prompt 唯一性、answer/alias 和原始 distractor；必要时提出受限 revision | `codex_adjudicated` 不等于 `human_gold` |
| 静态生成 | `qwen3.8-max` translation 与 distractor/context 自动检查、`gpt-5.5` translation review/备用 distractor、`gpt-5.6-sol` Targeted/Neutral generation | 生成或初审双语材料，不读取代理行为 | 生成器或自动 judge 不能批准最终 freeze |
| 静态终审 | 行为盲态 Codex | 审核最终渲染的 translation、alias、D1/D2、每个 Neutral/Targeted，并选择冻结 candidate | 不能预测行为模型一定会 flip |
| 代理行为 | 下述冻结四模型 | 只测量 Development 上的 MCQ 行为并产生逐模型 proxy 标签 | 不能改写静态材料，不能提供 exact-HF/PNT 机制证据 |
| exact-HF | 后续唯一 checkpoint | 完整重跑、开放补全、hidden-state、vector、intervention 与 repair | 在身份冻结前不得由代理结果替代 |

自动 reviewer 只提供中间证据；最终静态接受必须经过 Codex 逐项硬检查。若自动 reviewer 与 Codex 冲突，项目进入
`revise/reject/defer`，不得用投票硬过。任何行为输出都不能反向修改事实真值、翻译或 distractor 质量标签。

`G0A` 内部按以下顺序闭合，但三者共同通过后才算 `G0A_STATIC_CONTENT_POOL_FROZEN`：

```text
G0A.1 SOURCE_BOUND_AND_FACT_ADJUDICATED
  → G0A.2 STATIC_GENERATED_AND_AUTOMATICALLY_REVIEWED
  → G0A.3 CODEX_SEMANTICS_AND_LEAKAGE_FROZEN
```

### 3.1 独立统计单位

主统计单位固定为：

```text
base_fact_id
```

同一事实的多个 distractor、多个 perturbation、两种语言、多个 arm、多个模型、多个层或 scale
均是嵌套测量，不能分别计为独立缺陷。

行为 variant 使用：

```text
weakness_variant_id = hash(
  base_fact_id,
  designated_distractor_id,
  static_content_pool_sha256,
  behavior_model_manifest_sha256
)
```

每个 `base_fact_id` 的事实、翻译、alias、distractor、扰动变体和所有测量必须继承同一 split。
同一 leakage component 不得跨 split。

### 3.2 双 distractor variant

每条事实固定一个三选一 MCQ：

```text
options = [gold, distractor_1, distractor_2]
```

选项顺序由 `base_fact_id + protocol_version` 的冻结哈希确定。两个 distractor 都必须在目标模型输出出现前通过审核并冻结，
且分别构造一对上下文：

```text
variant D1: Neutral_D1 + Targeted_D1(designated=distractor_1)
variant D2: Neutral_D2 + Targeted_D2(designated=distractor_2)
```

`Neutral_D1` 与 `Neutral_D2` 必须分别匹配各自 Targeted 的句式、长度和语体；不得用一个通用 Neutral 代替两者。
EN/ZH Original 在两个 variant 之间共享并只执行一次。于是每个事实、每个行为模型有 10 个唯一输入：

```text
EN/ZH Original                                      = 2
EN/ZH Neutral_D1 + EN/ZH Targeted_D1               = 4
EN/ZH Neutral_D2 + EN/ZH Targeted_D2               = 4
总计                                                 = 10
```

标签按 `(base_fact_id, distractor_id, model_id)` 独立产生；汇总时仍以 `base_fact_id` 为独立单位，报告
`0/1/2` 个成功 variant、`any_success` 和 `both_success`。两个 variant 均成功不能把一个事实计成两个独立样本。
进入向量或回归分析时，必须先在事实内平均，或令每个成功 variant 的权重为
`1 / 该事实的成功 variant 数`。

Development、当前及后续 exact-HF 重跑，以及选择 `induced_confirmation` 时的 Validation/Sealed 均统一运行两个冻结 variant；
不得查看结果后只保留翻转更强的 distractor。

### 3.3 exact-HF 模型身份

PNT 主链只绑定一个 exact-HF model manifest，至少记录：

- HF repository ID 与 immutable checkpoint revision；
- tokenizer ID/revision；
- 权重精度或量化方式；
- prompt/chat template 及其 SHA；
- decoding、answer parsing 与 scoring policy；
- answer-tokenization policy；
- hook 名称、layer mapping 和 hidden-state 读取位置。

同一身份必须贯穿：

```text
MCQ behavior
→ 自然 EN/ZH open-completion baseline
→ hidden states / Logit Lens
→ task vector / difference vector
→ intervention
→ Validation / Sealed / regression controls
```

API/Ollama/panel 只允许在 `P0_DEV_PROXY_SCREEN` 中作为行为筛查；其信号必须逐模型单独报告，
不能并入 exact-HF 的机制样本，也没有过滤随后完成的 exact-HF 全量重跑集合。

## 4. MCQ 双 variant 与分层标签

每个 `(base_fact_id, designated_distractor_id, model_id)` 在逻辑上有六个条件：

```text
EN Original / Neutral / Targeted
ZH Original / Neutral / Targeted
```

两个 distractor 共享 EN/ZH Original，所以每个事实物理执行 10 个而不是 12 个输入。
两级标签是嵌套标签，不得伪装成互斥类别。

### 4.1 代理高召回标签 `proxy_directed_candidate@model_id`

对指定 distractor 必须同时满足：

```text
ZH Original  = gold
ZH Neutral_d = gold
ZH Targeted_d = designated distractor
```

英文结果不作为这一层的准入条件，但必须完整保存，用于下一层分类。
该标签用于高召回挖掘定向脆弱性，不能写成 `HF defect`、`PNT eligible` 或已验证修复对象。

### 4.2 代理中文特异标签 `proxy_zh_specific_strict@model_id`

必须先满足 `proxy_directed_candidate@model_id`，再同时满足：

```text
EN Original  = gold
EN Neutral_d = gold
EN Targeted_d = gold
```

这保留了原严格规则的中文特异含义，但仍只是相应代理模型、相应路由和相应运行时刻的行为证据。
四个代理模型各自出标签，不作投票，不把任一模型阳性的 union 当成共享正式标签。

这里“放宽 strict”的准确含义是：进入缺陷挖掘漏斗和未来 MCQ-conditioned PNT 的主 gate，
由旧的六臂全正确条件放宽为 `directed_candidate` 的中文三臂条件；英文三臂不再否决一个方向明确、
Neutral 稳定的中文定向缺陷，只用于判定更窄的 `zh_specific_strict` 和英文保持性。
这不是放宽 Targeted 的指定命中、删除 Neutral，或用多模型投票补足单模型证据。

### 4.3 exact-HF 重新确认标签

活动 exact-HF 已在完整 Development 96 上运行全部 10 个输入，并从自己的输出重新产生；
后续新协议版本仍必须遵守同一要求：

```text
hf_directed_candidate:
  ZH Original = gold
  ZH Neutral_d = gold
  ZH Targeted_d = designated distractor

hf_zh_specific_strict:
  hf_directed_candidate
  + EN Original/Neutral_d/Targeted_d = gold
```

只有 `hf_directed_candidate` 才可进入 `MCQ-conditioned PNT` 的归因/修复；
跨语言“中文特异”主张必须进一步限定在 `hf_zh_specific_strict`。
代理阳性不能直接进入任何 hidden-state、vector 或 intervention 分析。

### 4.4 exact-HF 抗扰动对照 `hf_resistant_control`

```text
EN Original  = gold
EN Neutral_d = gold
EN Targeted_d = gold
ZH Original  = gold
ZH Neutral_d = gold
ZH Targeted_d = gold
```

它接受了相同类型的 Targeted 操纵但仍保持正确，是机制比较的主要对照。

### 4.5 其他状态

| 标签 | 定义 | 用途 |
|---|---|---|
| `baseline_failure` | 相应分析要求的 Original 已错 | 不进入对应 defect/control 主比较 |
| `neutral_unstable` | Original 正确但 Neutral 失败 | 说明一般上下文不稳定，单列 |
| `off_target_failure` | Targeted 错但未命中指定 distractor | 非定向错误，单列 |
| `shared_language_susceptibility` | EN Targeted 也命中指定 distractor | 不是中文特异缺陷，单列 |
| `incomplete` | 任一必要 arm 非 terminal 或缺失 | 基础设施/缺失报告 |
| `weak_continuous_signal` | 未 flip，但 correct-vs-distractor margin 明显下降 | 仅作预声明的连续效应分析 |

不得通过删除 Neutral、放宽 designated-distractor hit、重试不理想答案，或用 panel 投票替代单模型合同来增加候选数量。

## 5. 匹配对照合同

PNT 不能只分析成功被扰动的正例。对照必须在查看 hidden state 和干预结果前构造。

硬条件：

- 同一 split；
- 同一 exact-HF 模型；
- 同一 `probe_relation_id`；
- 同一 answer type；
- 不属于同一 leakage component；
- 相应 variant 满足 `hf_resistant_control`。

优先匹配变量：

- answer token length；
- prompt tier/template；
- Original/Neutral 的正确答案 margin；
- prompt/context 长度；
- distractor 类型、长度与审核质量。

默认保留全部合格 resistant controls，并用预声明匹配权重或协变量调整；
若使用 1:1 匹配，算法、caliper 与 tie-break 必须在 Development 冻结。
无合格对照的 `hf_directed_candidate` 标为 `unmatched`，保留在漏斗分母中，但只作描述性个案。

## 6. PNT 自然基线与结论分层

exact-HF 重新确认的 `hf_directed_candidate` 可以进入“诱导弱点”的机制研究，但自然 EN/ZH open completion
仍是 PNT 的必备基线，不能用 MCQ Original 替代。directed candidates 与 resistant controls 都必须运行
同一 exact-HF 的自然基线。

按自然基线另行标注：

| 标签 | 自然 open completion | 可支持的结论 |
|---|---|---|
| `natural_pnt_gap` | EN 正确、ZH 错误或显著 margin gap | 自然跨语言事实召回弱点 |
| `induced_only_gap` | 自然 EN/ZH 均正确，但 MCQ Targeted 出现 `hf_directed_candidate` | MCQ 条件下的诱导脆弱性 |
| `natural_en_failure` | EN 已错误 | 不进入跨语言 PNT 主分析 |

Targeted context 去掉选项后的 option-free completion 可作为预声明的桥接测量：

- 若也生成指定 distractor，可报告 `targeted_open_transfer`；
- 若不转移，MCQ 缺陷仍可用于 `MCQ-conditioned PNT`，但不得写成自然事实召回缺陷；
- 不得把上述两类样本合并成一个更强的 PNT 结论。

## 7. 分阶段操作与 Gate

### Stage 0A：全 160 静态内容池生成、Codex 审核与冻结

#### 输入

- 唯一的 v3 semantic-aware post-closure pre-perturbation bundle；
- 160 个 facts/prompts；
- 待生成/审核的 EN→ZH translation 与 answer aliases；
- 320 个 split-local distractor candidates；
- duplicate/leakage closure 与 historical-exposure 证据。

#### 操作

1. 以来源证据和确定性规则检查事实、prompt tier、答案、ID、SHA 与 lineage。
2. 由 Codex 逐项裁决事实、translation、alias 和两个 distractor；审核时不得查看四个代理模型或目标模型的行为结果。
3. 对每个 distractor 分别生成 `Neutral_d` / `Targeted_d`，并由 Codex 逐项审核事实性、答案唯一性、关系保持、
   translation equivalence、Neutral neutrality 和 Targeted direction。
4. 完成 semantic-paraphrase closure 与 historical-exposure 判定。
5. 冻结 96/32/32，所有关联变体跟随 `base_fact_id` / leakage component。
6. 冻结双 distractor 候选池、成对 context、三选一选项与顺序、缺失/重试策略；生成每条 10 个模型无关的静态输入规格。
7. 为 Development 生成行为盲态交叉拟合 fold；不得使用任何代理/目标模型输出。原设计要求在 P0 前落盘，
   实际产物的时序限制见本节 gate 后的补充说明。

这里对全 160 做静态生成与 Codex 审核，不读取行为输出，不算消费 Validation/Sealed。
Codex 替代人工终局审核时必须如实记录。当前 `static-g0a-codex-decision-v1`
产物使用的实际字段是：

```text
reviewer_type = codex_proxy
decision = accept | revise | reject | defer
terminal_status = completed  # 仅 terminal 记录
human_gold = false
```

若导出到通用审核 schema，可以映射为 `adjudicator_type=codex`、
`review_status=codex_adjudicated`，但不得声称当前 JSONL 原生含有这两个字段。
当前 `review-v5` 已保存 rubric、reason、review-input binding 与 decision bundle SHA，但未独立证明
Codex model/version、prompt SHA 或原始 output SHA；不得声称这些 provenance 字段已经存在。
若后续补充，必须通过单独 manifest 明确绑定。证据不足必须 `defer`，不能猜测通过。
Codex 裁决不是 `human_gold`，也不得写成独立人工验证。

任何依赖代理/目标模型行为的生成、选择或规则调整都只能发生在 Development。
若 `P0` 后修改任何静态内容，`G0A` 自动失效，必须发布新版本并重新执行整个 `P0`；
不得只重跑或保留表现更好的 variant。

#### `G0A_STATIC_CONTENT_POOL_FROZEN`

协议 gate 名 `G0A_STATIC_CONTENT_POOL_FROZEN` 对应机器可验的
`status=g0a_static_stimulus_frozen`，不得仅凭静态文件存在或 review 完成视为通过。

- 160/160 有 terminal 的来源规则检查和 Codex 裁决；所有 accepted 项保持 `human_gold=false`；
- translation/alias 逐项审核完成；
- 320/320 distractor candidates 有 terminal review，且每条两个候选均可用；
- 每个 distractor 都有独立配对的 EN/ZH Neutral/Targeted，160 条共 1,600 个模型无关静态输入规格；
- semantic-paraphrase closure 与 historical-exposure inventory 已完成；若无法关闭对应 blocker，则 G0A 失败；
- historical exposure 只有在 manifest 为 `status=complete` 时才算通过；必须绑定当前工作树、全部 reachable
  Git blobs 的扫描证据，以及 scope owner 对外部目录、另行脚本运行和外部平台历史的显式 attestation；
  仓库内零命中不能替代仓库外声明；
- `review_freeze_manifest`、`split_freeze_manifest`、Codex adjudication manifest 与 `static_freeze_manifest`
  齐全并绑定 SHA；
- 若主张 relation-conditioned PNT，四个 `probe_relation_id` 已逐项审核并冻结；
- `static_frozen=true`，但 `behavior_authorized=false`、`exact_hf_render_frozen=false`、
  `pnt_authorized=false`；
- Validation/Sealed 仍无目标模型行为输出。

历史时序说明：`development_fold_manifest.json` 与 `perturbation_protocol_manifest.json` 未在已完成的
P0 前落盘，现作为 `post_p0_pre_g0b` 补充冻结保存于
`static-g0a-160-dual-v8-20260916/pre-exact-hf-protocol-v1/`。二者不能追溯表述为 P0 前预注册，
也不改变既有 `frozen-v1/static_freeze_manifest.json` 或要求重跑 3,840-call P0。生成器只读取冻结的
G0A、split、静态 bundle 与本权威协议，不读取 proxy/HF 行为输出；因此该补充冻结仅对尚未开始的
exact-HF replay、PNT 与 repair 前瞻有效。所有 exact-HF 身份、渲染、tokenization、layer、scale、
threshold 等 G0B 字段仍为 pending，`pnt_authorized=false`。

未通过时只返回 Stage 0A。允许继续进行静态生成、确定性检查和 Codex 审核；
禁止任何代理/目标模型回答这些 MCQ，禁止 PNT 和“先跑一点看看”的行为实验。

### `P0_DEV_PROXY_SCREEN`：四模型 Development 代理筛查

只在 `G0A` 通过后、仅对 Development 96 执行。活动 v10 固定四个代理行为模型：

```text
qwen3.6-27b
gemini-3.7-flash
gemma3:12b
llama3.1:8b
```

活动 v10 将 `gemini-3.7-flash` 固定为 `max_output_tokens=512`。这是针对 v9 的
256-token `truncated_json` 失败发布的新 config/run version，不得回填或续跑 v9。

前两个按当前配置属于 API/gateway 路由，后两个属于本地 Ollama；逻辑 profile 名不能被表述成已验证的官方直连 provider。
正式批量前必须逐模型做代表性 route/runtime probe，并冻结可获得的模型、路由、模板、参数、时间和请求/响应哈希；
路由失败记 `incomplete`，不得在同一 run 临时换模型。

`proxy_representative_probe_manifest.json` 必须绑定四个 route/runtime identities。托管路由只保存
规范化 route/base URL 的 SHA，不保存 credential 或明文 endpoint；Ollama 绑定 server version、
installed model digest，以及可取得时的 template SHA，无法取得的字段必须记录为 `unknown` 并附 limitation。
每个 probe 和 full-run result 都必须绑定 identity SHA 并检查 response-model compatibility；summary
必须绑定完整 identity-set SHA。该合同只有在对应 provenance 修改和测试通过后才可写成已执行证据。

每个代理模型执行：

```text
96 facts × 10 unique inputs = 960 measurements
4 models × 960              = 3,840 measurements
```

这里的 3,840 是主实验的 logical behavior calls；正式批量前另有 4 个 representative probe calls
（每模型 1 个），不进入三臂分析。当前 `max_retries=1` 表示每个 logical call 最多产生 2 次 request
attempts；logical calls、attempts 与 retries 必须分别报告。

当前统一 runtime 已实现且只允许按下列顺序进入代理阶段：

```text
prepare-static-proxy-run  # 绑定 g0a_static_stimulus_frozen manifest，仅展开 Development 96
→ probe-static-proxy-run  # 四模型各一次、同一确定性 stimulus 的 route/runtime probe
→ run                     # probe manifest 通过后才执行 3,840 次完整行为测量
→ audit
→ analyze-three-arm
→ report
```

入口可用本身不代表已获执行资格；在本节 gate 通过前，连 `prepare-static-proxy-run`
也不得伪造 frozen manifest，`probe-static-proxy-run` 与 `run` 更不得提前调用。当前 v10 已通过
G0A 与四模型 probe，并完成 `run → audit → analyze-three-arm → report`；该完成态只成立于
Development proxy behavior 层，不解锁 exact-HF 或 PNT。

每个模型独立生成 `proxy_directed_candidate@model_id` 与 `proxy_zh_specific_strict@model_id`，
同时保留 neutral instability、off-target、shared-language、incomplete 和连续 margin 信号。
不得 panel 投票、不得把四模型 union 当作正式标签，也不得用行为结果替换 distractor 或改写静态材料。

`P0_PROXY_REPORT` 至少确认：

- 四模型分别报告 96 个事实、192 个 variant 的完整漏斗与缺失原因；
- 原始请求、响应、解析、retry、模型/路由 provenance 与哈希可追溯；
- Validation/Sealed 的 `proxy_behavior_exposed=false`；
- 代理结果只写入比较报告和 `hf_retest_priority` 元数据，不改变未来 exact-HF 的重跑集合；
- `pnt_eligible=false`，不采集 hidden states，不构造 vector，不运行 intervention。

当前 v10 的实际代理结果如下。variant 计数是同一 `base_fact_id` 内的嵌套描述量，不能当作独立样本量：

| 代理模型 | completed calls | `proxy_directed_candidate` variants | `proxy_zh_specific_strict` variants |
|---|---:|---:|---:|
| `qwen3.6-27b` | 960 | 55 | 28 |
| `gemini-3.7-flash` | 960 | 168 | 102 |
| `gemma3:12b` | 960 | 91 | 44 |
| `llama3.1:8b` | 960 | 41 | 14 |

全局审计为 3,840/3,840、Development=3,840、Validation/Sealed=0、
`proxy_screen_complete=true`、`operational_reasons=[]`、`panel_union_reported=false`。
`gate_passed=false` 的唯一原因是 `proxy_behavior_only_not_pnt_eligible`；因此这些只能称为逐模型
proxy candidates，不能称为 exact-HF 缺陷、PNT 资格或机制证据。

### Stage 0B：exact-HF 身份与最终渲染冻结

**执行状态（2026-09-19）：已完成。** 权威目录为 `exact-hf-qwen3-8b-g0b-v3/`；运行冻结为
batch size 1（每个 MCQ 内三条 completion 同批评分）、eager attention、deterministic algorithms、
TF32 关闭。

exact-HF 可用后，基于未被代理行为修改的 `G0A` 静态内容池执行：

1. 冻结 HF repository/revision、tokenizer、精度、chat template、decoding、parser、scoring 与 answer-tokenization policy。
2. 将每条事实的 10 个静态输入规格渲染成 exact-HF 最终输入，并审计 token 长度、选项偏差和渲染 SHA。
3. 冻结 hook 名称、layer mapping、hidden-state 位置以及 missing/retry policy。
4. 在 exact-HF 行为运行前，依据补充协议 manifest 解析并冻结仍为 pending 的 G0B 字段；不得按代理
   阳性/阴性选择样本。

#### `G0B_EXACT_HF_RENDER_FROZEN`

- exact-HF model manifest 与最终 render manifest 齐全并绑定 `G0A` SHA；
- 全 160 的 1,600 个 exact-HF 输入均可渲染、可解析且 token audit terminal；此时只开放 Development 96 的 960 个输入执行；
- 96 个事实全部进入 mandatory exact-HF replay queue，包括四模型全阴性的事实；
- proxy 标签只作为事后迁移比较字段，不参与准入、选项、context 或重试策略；
- Validation/Sealed 仍无行为输出，`pnt_authorized=false`。

### Stage 1：exact-HF 完整重跑 Development 96

**执行状态（2026-09-19）：已完成。** 权威目录为 `exact-hf-qwen3-8b-development-v3/`；960 条结果
独立复跑 SHA 一致。最终保留 96 个 directed variants/57 facts、14 个 ZH-specific variants/11 facts、
21 个无 final-score 精确并列的 resistant variants；其中 44 个 directed variants 有同关系、同答案类型
对照，52 个没有匹配对照。

#### 操作

1. 仅在 Development 96 上运行每条 10 个 exact-HF MCQ 输入；共享 Original 不重复请求。
2. 先处理缺失与基础设施失败，再产生 terminal failure/state 分类和嵌套 flags；
   `hf_zh_specific_strict` 必须是 `hf_directed_candidate` 的子集。
3. 按 variant 产生 `hf_directed_candidate`、`hf_zh_specific_strict`、`hf_resistant_control`、off-target、
   neutral-unstable 和 invalid，再以 `base_fact_id` 等权汇总。
4. 在不查看 hidden state/repair 结果的前提下冻结匹配策略。
5. 如需调整 context/selection/label 规则，只能发布新的 Development protocol version；旧结果不得混合。
6. 将最终规则盲态应用到 Validation/Sealed 的双 variant 静态池；不得选择代理或 exact-HF 上更易翻转的一个。

exact-HF 主行为测量规模为：

```text
96 facts × 10 unique inputs = 960 measurements
```

必须重跑全部 96，而不是只重跑任一代理模型阳性的事实。

#### `G1_BEHAVIOR_PROTOCOL_FROZEN`

- 96 条、192 个 variant 按预注册 missing/retry policy 达到 terminal；
- exact-HF label、双 variant、matching、prompt/scoring 合同冻结；
- Validation/Sealed 没有被用于调参；
- 报告 unique `base_fact_id` 和全部漏斗分母。

不设置“必须 20%”“必须 30–50 条”或“四个关系都必须有 `hf_directed_candidate`”的 gate。
若 exact-HF directed candidate 为 0，停止诱导 PNT 主链，回到新的 Development protocol；不得把 proxy 阳性顶替进来。
若数量很少，只能做个案/探索性机制分析，不能声称稳定群体机制。

### Stage 2：Development 96 PNT 方法开发

**部分执行状态（2026-09-19）：** 自然 baseline v4 与 Codex proxy 语义复核已完成；MCQ
decision-position attribution v2 已完成 960 条 × 36 层，并将 `[960,36,4096]` BF16 `resid_pre`
保存到 `qwen3-8b-pnt-development-v1/mcq-decision-attribution-v2/`。正式运行与独立复跑的
metrics/index/activation SHA 全部一致，行为 final-token rank/order 差异均为 0。旧 v1 因未复刻
三 completion 行为 batch 形状而出现 62 个 rank 差异、29 个 order 差异，已降级为非权威诊断产物。

该完成态只证明 MCQ 选择决策位置的可复现内部测量，不证明原始 PNT 的 answer-text recall 或
EN→ZH translation 路径。vector intervention 继续 fail-closed，直到 option-free 三臂 prompt、显式翻译
prompt、两个 vector estimator 与四折 OOF 搜索/端点合同全部冻结。

HF 暂停后，已先完成不依赖模型推理的 G2A 文本级合同，路径为
`qwen3-8b-pnt-development-v1/g2a-option-free-preparation-v1/`：

- `option_free_three_arm_inputs.jsonl`：960 条，仅 Development；EN/ZH Original 各 96，
  EN/ZH Neutral 与 Targeted 各 192；不包含 MCQ 选项；
- `vector_source_prompt_specs.jsonl`：1,920 条；四个 OOF source partition 各 360，最终 Development
  全量 source partition 为 480；每个 source fact 均包含 EN/ZH 5-shot recall、EN→ZH translation 与
  EN/ZH zero-shot recall；
- task vector 固定为每个事实先平均 EN/ZH 5-shot state，再跨 source facts 平均；difference vector 固定为
  EN→ZH translation mean 减去 EN/ZH zero-shot recall 的 fact-balanced mean；
- 5-shot donor 由冻结哈希排序，只能来自同关系、不同 leakage component，并且不得来自当次 evaluation fold；
- 离线审计已通过 960/1,920 数量和 SHA binding、四折 source/evaluation 隔离、donor 规则、Original 与
  natural-baseline 用户内容一致、Neutral/Targeted context-only 模板以及 alias-group 不相交检查。

该包状态仍为 `offline_contract_prepared_pending_exact_tokenizer_audit`。它不包含新 tokenization、hidden
state、vector 或 intervention，且 `validation_facts_used=0`、`sealed_facts_used=0`。恢复同一冻结 HF
checkpoint/tokenizer 后，必须先做 exact render、长度与 answer-prefix 审计。自然语义复核的 3 个 `defer`
已在合同中固定为二元 natural-gap 与 repair-rule selection 的缺失值，同时仍可保留在 MCQ-conditioned
分析中。

behavior-blind unrelated-fact control 也已离线冻结：从既有 `path_not_token/chinese-v1` 的全部 13 条
双语、prompt-ready、Codex 翻译审核事实中，不读取当前 Qwen3 行为地形成 26 个 EN/ZH 输入。它们与当前
160 cohort 无 source/prompt/answer 精确重叠，并经 bounded Codex 语义审核确认不是同一事实或释义；只作
retention regression，不进入正式 cohort、fold 或 vector source。由于只有 13 个独立事实，2% accuracy-drop
阈值在该面板上等价于不允许新增受损事实，必须逐项报告，且不得据此声称总体安全性。其 tokenizer render
与 no-op baseline 仍待 HF 恢复后执行。至此剩余离线合同 blocker 已关闭，但 Development intervention
仍须等待 exact-tokenizer 审计。

#### 交叉拟合

原设计要求在 Stage 0A 将 Development 按 relation 与 leakage component 预先分折；实际
`development_fold_manifest.json` 是 P0 后、G0B 前的补充冻结，不构成 P0 前预注册。由于其生成仅使用
G0A 静态字段且不读取 proxy/HF 行为输出，它可前瞻用于尚未开始的 exact-HF/PNT 交叉拟合，但报告中
必须持续披露这一时序限制。当前 4 folds 均为 24 条、每个 `probe_relation_id` 精确 6 条，且 leakage
component 不跨折；组件原子性始终优先于等额目标。

对每一折：

```text
其余 72 条：只构造 recall task vector / translation-minus-recall difference vector
留出 24 条：只评估预声明的 layer / scale / vector combination 网格
```

每折 vector-source 使用全部符合预声明训练资格的事实，不得按该折的
`hf_directed_candidate` / `hf_resistant_control` 结果事后挑选。
汇总四折 out-of-fold 结果后，才按预声明规则作一次全局选择，冻结 layer、scale、vector type、trigger 和 repair rule；
再用全部 Development 96 重建一次最终 vector，供 Validation/Sealed 使用。
任何一条 Development 记录在 out-of-fold 评估时，都不能被用于产生当次注入的 vector。

#### 必做分析

自然 EN/ZH open completion 对全部 Development 96 运行；对 `hf_directed_candidate` 与
`hf_resistant_control` 还要同时运行：

- 两个 variant 的 MCQ Original/Neutral/Targeted hidden states 与 Logit Lens；
- `no_intervention`；
- `norm_matched_random`；
- `task_only`；
- `difference_only`；
- `combined`；
- English retention 与 unrelated-fact regression。

最低报告指标：

- alias-aware generation correctness；
- complete-answer sequence log probability；
- answer-token rank；
- `logP(gold)-logP(distractor)` margin；
- 每层 hidden-state / Logit-Lens 指标；
- intervention 前后的 paired change；
- control harm rate。

机制的主对比建议为：

```text
[Targeted − Neutral]_hf_directed − [Targeted − Neutral]_hf_resistant
```

配置选择顺序固定为：

1. 先满足 Original、Neutral、英文、resistant controls 和 unrelated facts 的保持约束；
2. 再最大化 `hf_directed_candidate` 上 ZH Targeted 的 paired margin 改善；
3. 必须优于 no-intervention 与 norm-matched random；
4. 并列时选择最小 `abs(scale)`，再按冻结 layer 顺序选择。

#### `G2_REPAIR_RULE_FROZEN`

- 四折均无训练/评估重叠；
- vector source IDs、vector 文件与 SHA 完整；
- model、prompt、scoring、layer、scale、vector combination、trigger、endpoint、保持阈值和 tie-break 全部冻结；
- Validation/Sealed 对上述选择没有贡献；
- 最终 vector 只由 Development 96 重建。

### Stage 3：Validation 32 固定确认

第一阶段不运行 Validation；只有 `G2` 通过后才开启。

在开启前必须预声明二选一：

#### A. `induced_confirmation`（推荐）

- 对全部 32 条运行冻结的双 variant MCQ（每条 10 个唯一输入，共 320 个）和自然 EN/ZH baseline；
- 在预先选定的层/位置采集冻结的 hidden-state / Logit-Lens 指标，不再搜索层；
- 不先按结果挑样本，统一运行冻结 intervention/no-op/random/control；
- 之后再按冻结规则报告 `hf_directed_candidate`、`hf_zh_specific_strict` 与 `hf_resistant_control` 条件结果。

该模式验证“固定扰动所定义的缺陷表型 + 固定 PNT 修复”。它不是在主张扰动生成器对所有关系普遍泛化。

#### B. `natural_only_confirmation`

- 不运行 MCQ 扰动，只运行自然 PNT 与 repair；
- 只能验证自然事实召回机制；
- 不得声称诱导缺陷修复在 Validation 得到确认。

#### 禁止

- 重生或替换 distractor；
- 修改 exact-HF label/matching 规则；
- 重算 vector；
- 调整 layer/scale/threshold/endpoint；
- 因某关系无 directed candidate 而补样、搬 Development 正例或只保留两个 distractor 中更强的 variant。

#### `G3_VALIDATION_CONFIRMED`

若选择模式 A：

- 全 32 条的主要连续效应方向符合预注册；
- 冻结的 directed-vs-resistant attribution contrast 方向得到复现；
- `hf_directed_candidate` 子集的 repair 改善优于 no-op/random，且 resistant control harm 在容忍度内；
- Original、Neutral、英文和 unrelated facts 通过保持阈值；
- 同时报告全体 32 条 estimand、directed 条件 estimand 和中文特异子集。

若 directed 事件少于 protocol manifest 中预先基于功效/精度设定的最小事件数，结果只能写“证据不足”，
不得放宽规则。若需继续，新增独立确认 cohort。

如果 Development 的 directed 率约为 20%，32 条 Validation 只期望约 6–7 个事件，
因此关系级结果默认只作描述，主要补充终点应使用全体合格事实的连续 margin 变化。

若 Validation 后修改任何规则，原 Validation 自动降级为 Development；必须另取新的 Validation/Sealed。

### Stage 4：Sealed 32 最终一次性评估

仅在 `G3` 通过、最终 vector 与 repair manifest 冻结后开启。

- 采用与 Validation 相同且预先冻结的 A 或 B 模式；
- 除预注册基础设施重试外只开启一次；
- 对全部 32 条运行统一流程，不能先筛 directed candidate 再决定运行哪些干预；
- 输出全体结果、directed 子集、中文特异子集、resistant 子集、匹配结果和每关系描述；
- 运行英文保持和无关事实回归。

#### `G4_FINAL_REPORT`

只有在目标改善超过 no-op/random，且保持与回归均通过时，才能称为 `validated repair`。
否则报告为 `exploratory intervention behavior` 或 `insufficient evidence`，不得调参重跑并覆盖第一次结果。

### Stage 5：2,450 条扩量（当前阻塞）

扩量不属于 160 条闭环。只有满足以下条件后才重新设计并另行授权：

- Stage 4 完成；
- exact-HF MCQ directed/zh-specific labels、PNT attribution 与 repair 的产物可追溯；
- exact-HF 小闭环显示有足够信息价值；
- 新数据完成统一 schema、ID、semantic/leakage closure 与新的 component split；
- 扩量目标明确是提高缺陷发现量，而不是补救失败的独立验证。

不得用 broad `mapped_relation` 作为全部扰动数据的准入门槛；
只有 relation-conditioned PNT 才要求经过审核并冻结的 `probe_relation_id`。

## 8. 固定报告分母与主指标

代理阶段对四个模型分别报告，不能合并分母：

```text
N_assigned_facts = 96
N_assigned_variants = 192
→ N_10_input_terminal_facts
→ N_evaluable_variants
→ N_proxy_directed_candidate_variants
→ N_proxy_zh_specific_strict_variants
→ N_neutral_unstable / N_off_target / N_shared_language / N_incomplete
```

exact-HF 阶段按 split 报告：

```text
N_assigned_facts
N_assigned_variants = 2 × N_assigned_facts
→ N_10_input_terminal_facts
→ N_hf_directed_candidate_variants / N_hf_zh_specific_strict_variants
→ N_hf_resistant_variants / N_off_target / N_other_invalid
→ N_directed_facts(any/both) / N_matched_facts / N_unmatched_facts
```

variant 数用于描述两种扰动结果；效应估计、置信区间与独立样本量以唯一 `base_fact_id` 为单位，
并按预声明的事实内平均/权重处理两个 variant。失败、缺失、重试和排除原因不得静默删除。
比例同时给出 `n/N` 与 95% CI。

主结果分三层：

1. 代理筛查：每模型分别报告 directed 与 zh-specific 的 variant/fact 级发现率；
2. exact-HF 条件修复：`hf_directed_candidate` 上 fixed intervention 相对 no-op/random 的 paired 改善；
3. 全体连续效应：所有合格事实上的 correct-vs-distractor margin 变化。

关系分层在每格样本很小时只作描述，不能据此声称关系级普遍泛化。

## 9. 必须产物

以下是 artifact contract。当前代理阶段已有统一 runtime；后续 exact-HF/PNT 产物仍不能仅凭清单
推定对应实现、模型身份或实验结果已经存在：

```text
review_freeze_manifest.json
split_freeze_manifest.json
codex_adjudication_results.jsonl
codex_adjudicator_manifest.json
static_content_pool.jsonl
static_content_pool_manifest.json
development_fold_manifest.json
perturbation_protocol_manifest.json

run_manifest.json
proxy_behavior_inputs.jsonl
proxy_representative_probe_results.jsonl
proxy_representative_probe_manifest.json
simulation_results.jsonl
preholdout_summary.json
three_arm_analysis.json
experiment_report.md

exact_hf_model_manifest.json
exact_hf_render_manifest.json
hf_replay_queue.jsonl

exact_hf_development_behavior_results.jsonl
exact_hf_defect_control_labels.jsonl
matching_manifest.json
natural_open_completion_baseline.jsonl
natural_semantic_review_results.jsonl
natural_semantic_review_manifest.json

mcq_decision_logit_lens.jsonl
mcq_decision_activation_index.jsonl
mcq_decision_resid_pre.pt
mcq_decision_attribution_manifest.json
attribution_runtime_stability_audit.json

vector_source_manifest.json
task_vector.safetensors
difference_vector.safetensors
development_oof_results.jsonl
repair_rule_freeze_manifest.json

validation_results.jsonl
validation_confirmation_report.json

sealed_open_manifest.json
sealed_results.jsonl
sealed_final_report.json
```

两个 P0 后/G0B 前补充冻结产物位于
`data_processed/factual_perturbation/static-g0a-160-dual-v8-20260916/pre-exact-hf-protocol-v1/`；
其中 `exact_hf_resolution_status=pending_g0b` 是生成时状态，不能单独用来判断当前执行进度。当前
exact-HF 身份与 Development MCQ attribution 已由后续 G0B/G1/attribution manifests 绑定；但仍不能
据此声称 answer-text vectors、intervention、repair、Validation 或 Sealed 已获授权。

每个 manifest 至少绑定：输入路径、schema/protocol version、row count、唯一 fact/variant ID 数、SHA-256、
适用的模型/路由身份、时间戳、生成代码版本和上游 manifest SHA。
静态 manifest 不得伪造尚不存在的 exact-HF identity；代理 manifest 和 exact-HF manifest 必须彼此独立。

## 10. 验收标准

### 数据与泄漏控制

- 160 个唯一 `base_fact_id`，顶层 96/32/32 保持不变；
- 每个事实有两个已审核 distractor、两对 Neutral/Targeted、共享 Original 和 10 个静态输入规格；
- 所有 variant 与 leakage component 不跨 split；
- Codex 裁决全部使用 `reviewer_type=codex_proxy`、terminal `decision`/`terminal_status` 和
  `human_gold=false`；若另行导出通用字段，映射必须显式且不得冒充人工金标；
- 四代理只接触 Development 96，Validation/Sealed 保持 `proxy_behavior_exposed=false`；
- Validation/Sealed 不进入 vector 构造或调参；
- exact-HF identity 全链一致；
- review/freeze、运行失败和所有重试均可追溯。

### 缺陷数据

- 代理的 `proxy_directed_candidate` 与 `proxy_zh_specific_strict` 按单模型、单 distractor 分层判定；
- Targeted 必须命中 designated distractor；
- neutral instability、off-target error 与 panel/union 结果不混入两级代理标签；
- 活动 exact-HF 已对全部 Development 96 重跑 960 个输入，并重新生成自己的标签；
- 只有 exact-HF directed defect 与 resistant control 进入机制分析；
- 一个事实的两个 variant 都保留，但统计独立样本量仍只计一次。

### 机制与修复

- 自然 EN/ZH baseline 齐全；
- Development 交叉拟合无样本自用；
- no-op、norm-matched random、task-only、difference-only、combined 齐全；
- Validation 只确认，Sealed 只作最终评估；
- 同时报修复、对照伤害、英文保持和无关事实回归。

### 正确的 Go/No-go

不再使用以下硬门槛：

- “MCQ 候选率必须 ≥20%”；
- “必须得到 30–50 条候选”；
- “四个关系都必须有 directed candidate”；
- “Validation/Sealed 必须复现某个候选率”。

真正的决策链是：

```text
缺陷是否方向明确且 Neutral 稳定？
            ↓
机制差异是否相对 resistant control 成立？
            ↓
固定干预是否在独立数据上优于 no-op/random？
            ↓
修复是否同时通过保持与回归检查？
```

候选数量只决定证据是确认性、探索性还是不足，不得反向改变缺陷定义。

## 11. 对旧设计冲突的最终取舍

| 冲突 | 取舍 |
|---|---|
| 从 160 随机抽 50 做 pilot | 取消；工程预检只能来自 Development，不能触碰 Validation/Sealed |
| 一个事实只用一个 distractor | 取消；两个 distractor 各自有 Targeted/配对 Neutral，Original 共享，每事实 10 个输入 |
| G0 同时要求静态内容和 exact-HF | 拆成 `G0A → P0 → G0B`；HF 不可用不阻塞静态冻结和 Development 代理筛查 |
| 一次运行全部 160 | 取消；四代理只跑 Development 96，exact-HF 也先完整重跑 96，再按 32 → 32 gate 开启 |
| MCQ flip 直接叫 PNT 候选 | 代理只产生 `proxy_directed_candidate` / `proxy_zh_specific_strict`；exact-HF 全量重测后才决定 PNT 资格 |
| 多模型 panel 共同决定 strict | 取消；四代理逐模型出标签，只有单一 exact-HF 形成机制链 |
| 用候选数量决定是否成功 | 取消；用方向性、对照差异、独立修复和保持/回归决定 |
| Validation/Sealed 是否跑扰动 | 首轮不跑；G2 前冻结模式。验证诱导修复时选 A 并各跑一次，natural-only 时选 B |
| 先扩到 2,450 再做 PNT | 取消；先完成 160 条 exact-HF attribution→repair 闭环 |

# Pitfalls → Paths Not Taken 联合研究设计评审

> 历史评审快照。2026-10-20新候选已改为“错误选项素材→开放三臂→复现缺陷”，MCQ不再作为准入门槛；
> 反馈限首次生成+最多一次修订。活动入口见[开放缺陷生成协议](../../tasks/task_open_factual_discovery_protocol.md)。
> 本文的“桥接未执行”等状态截至10月9日；10月16–17日结果见[tasks状态](../../tasks/README.md)。

日期：2026-10-09。依据：两个本地项目代码、当前任务协议、已保存的逐条结果与 manifest，以及 PNT 原论文。用户已明确：**以开放事实召回的机制与修复为主，MCQ 用于挖掘。**

本次进行了代码阅读、逐条数据关联、部分关键产物 SHA 核验；没有运行新的模型实验，没有修改既有 task 或冻结协议。计数与核验记录见同目录 `evidence.json`。用户随后明确：优先有效生成便于归因和修复的缺陷数据，不预先限制真实误导或错误断言等扰动家族；家族作为可解释的元数据和分析分层，不作为统一准入门槛。

## 1. 总体判断

研究方向可行，已有工作值得保留；当前最大的短板是**测量对象尚未完全接通**，并非原始数据不够。

Pitfalls 已提供事实、双语材料、扰动、三臂行为和溯源基础。PNT 提供事实回忆/目标语言输出的假设、内部测量与向量干预方法。但目前已完成的归因主要针对 MCQ 的 A/B/C 决策，尚未证明相同扰动在开放回答中造成同一种错误，更未完成开放回答上的因果修复和独立验证。

因此，后续任务应从“继续扩大经过审核的候选池”转为“在同一模型和输入形式上完成开放召回闭环”。160 条可作为已有基础上的机制可行性实验；五语言共同池用于后续规模化与外部确认。这个建议不否定此前按用户要求暂停 HF、优先翻译的安排，而是针对现在明确的研究终点重新排序。

## 2. 已完成什么，尚缺什么

| 对象 | 当前证据 | 对主目标的意义 |
|---|---|---|
| 全量公开事实池 | 8,969 起点，经审核和投影到 7,950，再到中文接受 7,663 | 事实与多语候选资源；不能直接称为模型缺陷集 |
| 五语言翻译 | 本地逐条文件 4,513 个事实，五语共同接受 3,326；共同集 dev/validation/sealed=1,983/673/670 | 已具备扩展资源，但仍缺最终 foil、匹配上下文、开放行为与机制资格 |
| 160 静态实验 | 96/32/32 划分，Development MCQ 960 条 | 可复用的先导实验与行为标签 |
| Qwen3-8B MCQ | 57 个事实至少一个 variant 定向翻转；11 个事实至少一个严格中文特异 variant | 是选择题表型，不等于开放召回表型 |
| 自然开放回答 | 已有 192 条，语义审核后 EN 36/96 正确、ZH 28/96 正确 | 已揭示 MCQ 与开放召回存在明显差异 |
| MCQ 内部测量 | attribution v2 指标及 index 文件可核验；报告完成 960×36 层 | 选择决策的可复现测量，尚不足以建立回答文本的回忆/翻译归因 |
| G2A | 960 个 option-free 文本规格、1,920 个 vector-source prompt 规格已存在 | 文本准备完成；tokenizer 审计、开放三臂执行、向量干预尚未完成 |
| 独立修复确认 | 未见该链 Validation/Sealed 修复结果 | 主目标尚未完成 |

本地 availability 需要单列：行为结果、缺陷标签、matching、归因指标和 activation index 的 SHA 与其 manifest 相符；当前 checkout 缺少 `frozen_static_bundle.jsonl`、G0B 的关键模型/渲染文件，以及 `mcq_decision_resid_pre.pt`。这不证明历史实验未执行，但意味着本机不能只凭现有 manifest 宣称可完整恢复。优先从原存储恢复并验 hash，不要为补文件而重跑所有 API 审核。

## 3. 最关键的科学问题

### 3.1 MCQ 阳性到开放召回缺陷的转换率尚未知

本次按 `base_fact_id` 联合现有 MCQ 标签与自然语义标签。57 个 MCQ 定向缺陷事实中：

| 当前自然状态 | 事实数 |
|---|---:|
| 自然英文失败 | 31 |
| `induced_only_gap`：自然英中均正确且有 MCQ 诱导缺陷 | 14 |
| `natural_pnt_gap` | 9 |
| 语义审核未完整 | 3 |

这说明“57 个 MCQ 缺陷”不能当作“57 个已知知识被扰动破坏的开放召回缺陷”。自然正确率低也不能直接推出模型没有该知识：prompt 形式、答案歧义和评分也可能影响结果，应通过固定的小样本错误审计区分，不应挑最好看的 prompt 回填旧结果。

**建议：对全部 Development 96 运行 option-free Original/Neutral/Targeted，保留 MCQ 阴性。** Original 与既有 natural baseline 完全一致、模型/评分也一致时可复用已有 192 条；剩余是 768 个上下文输入规格。统计 MCQ→open 迁移率，也统计 MCQ 阴性中的 open 缺陷，才能评价 MCQ 作为挖掘器的覆盖。

开放回答标签至少分为：

- `open_induced_directed`：目标语言无扰动和 Neutral 都答对，Targeted 生成指定错误答案。
- `open_induced_language_specific`：前述成立，并且英文三臂正确。
- `natural_language_gap`：英文自然正确、目标语言自然错误；用于自然弱点分支。
- `shared_susceptibility`、`baseline_failure`、`neutral_unstable`、`off_target`、`unresolved`：分别保留。

同一事实可在不同 variant 呈现不同状态，统计时先事实内聚合；不要强行把所有标签当成互斥集合。

### 3.2 当前 G2A 把生成迁移与候选打分迁移合在一起，必须拆开

现有 option-free 输入的合同将 `targeted_open_transfer` 写为 `generated_or_top_scored_designated_distractor_alias`。但给定几个候选时某候选得分最高，不代表自由生成会输出它。

本地已有一个直接例子：首条自然英文记录自由生成 `Iana Salazar`，正确答案是另一人；但 gold-minus-distractor 序列分数 margin 为正。相对两个指定错误答案更偏好 gold，仍不等于自由回答正确。

**主终点用 alias-aware/语义审核后的实际生成正确率和指定错误命中率；完整答案 log-probability margin 是连续辅助指标。** 分别报告 `generated_transfer` 与 `scored_preference_transfer`，不以 OR 合并。多 token 答案不能只用首 token rank 作为成功标准；别名聚合规则、生成长度和语义裁决方法需预先固定。

### 3.3 160 与五语言任务使用不同扰动家族

160 的冻结协议明确为 `explicit_false_assertion_v1`。G2A 中可见如“错误演员扮演某角色”的直接错误断言，随后要求补全同一事实。五语言任务则描述“真实干扰上下文”。两者分别测量知识与错误上下文冲突、以及事实保持下的显著性干扰，不能混合计算一个“扰动成功率”。

这也影响机制解释：显式错误上下文被模型采用，可能反映上下文采信、复制或任务理解，而非主要由回忆或翻译失败造成。应增加少量、预先定义的 Development 诊断：改变信息源表述、明确要求按真实世界知识回答、比较不同 cue 形式。诊断产生新协议版本，不能覆写现有冻结输入。

根据用户补充，取消预先指定某一扰动家族为主实验或压力测试的要求。真实误导、错误断言等都可作为候选生成策略，在 Development 中按开放缺陷有效产出、复现性、可解释性和成本选择。数据保存 `manipulation_family`，总体结果同时给出家族构成与分层结果，避免把由某一种操纵驱动的效果外推给所有策略。错误断言本身不是淘汰理由，但任务需要明确按现实事实而非虚构背景回答，且独立参考答案不能被扰动改写。

生成阶段的最低有效性要求是：基线与匹配 Neutral 能正确回答，Targeted 使实际开放生成出错，参考事实与问题仍可判定。指定 distractor 命中作为优先的定向缺陷标签；其他语义明确的错误另标为非定向缺陷，拒答和格式错误单列。机制解释尚未知不影响它成为行为缺陷候选；因果机制与修复效果留到后续验证。完整失败记录和对照保留，正例数据包可以是其中的一个视图。

工程上先用有界 Development 预算迭代“生成→快速有效性检查→同一目标模型开放三臂评测→标签与复现检查”，深入证据审核集中在拟纳入机制 cohort 的候选。生成器与静态裁决者分离，最终材料独立复核；优化生成策略可以使用 Development 行为反馈，但不能消费 Validation/Sealed 来持续调参。报告单位预算得到的独立有效事实数，以及全部被尝试的事实、候选和模型调用分母。不要仅以挑出正例后的成功率评价生成器，也不要把行为翻转直接称为已完成机制归因。

### 3.4 PNT 的机制假设需要在 Qwen3 上检验

PNT 原论文研究开放事实补全，用内部英文答案与目标语答案的层间轨迹，结合回忆 task vector 和 translation-minus-recall vector 分析与改善错误。这提供可检验的假设，不能直接当作 Qwen3 在当前扰动上的既定机制。[原论文](https://aclanthology.org/2025.emnlp-main.762/)

推荐先做同事实 Neutral→Targeted 的答案文本轨迹，观察 gold/distractor 与英/目标语形式的变化；英文自然正确只能提供基线证据，不能替代内部机制证据。若英文形式与目标语形式相同、首 token 重叠或答案是日期数字，应标记为语言转换诊断弱可辨识项，而不是把它们全部删除出行为集。

Logit Lens 与 hidden-state 差异属于描述性证据。增加定位后的 activation patching/ablation，并配不相关 donor、相同范数随机方向等对照，检验候选位置是否对错误产生必要或可恢复的影响。PNT 向量有效只能先证明这种干预有用；对特定计算阶段的归因还需阶段选择性和对照证据。若两种 PNT 向量都失败，应报告迁移边界，再检验上下文干扰等替代假设，不要强行套成翻译失败。

## 4. PNT 本地代码的接入评审

这些发现适用于当前本地 checkout，不推断原论文实验本身无效，也不意味着已经影响 Pitfalls 的独立 Qwen runner。

| 问题 | 代码证据 | 建议 |
|---|---|---|
| CLI 参数被覆盖 | `demo_intervention_performance.py:795` 解析参数后，798–819 硬编码 model、seed、layer、scale、condition、split | 接入前让实际 runtime 绑定参数；运行摘要写出真正使用值 |
| ICL ID 与实际 query 不一致 | `prepare_ICL_fv.py:48` 按原顺序保存 known_ids；67 随机选择 query；loader 在 `demo_load_datasets_model.py:264` 以 known_ids[i] 标记随机 query | query、答案、ID 必须来自同一条记录；否则 split 过滤不能证明隔离 |
| ICL donor 可含目标或 holdout | 同文件 77 从全 word_pairs 抽示例，没有排除 query；`save_task_vector:758` 在构造全池 ICL 后才过滤 train ID | 先传 source-fold 的记录，再从中选 donor，排除目标和同 leakage component |
| residual hook 不一致 | `compute_h_vector:741` 提取 resid_post；最终 evaluator 在 643 注入 resid_pre；估计 combined difference 时又在 288 附近使用 resid_post | 清晰固定层编号及 pre/post 坐标，no-op 与注入位置检查后再比较 |
| 模型/语言/关系写死 | loader 为既有模型分支；demo 固定六语和十种关系，difference 的 translation 语言列表也固定 | 使用当前 exact-HF Qwen 身份与统一 bundle adapter，不另行加载一个近似模型 |
| 开放评分较弱 | `demo_helpers.py:348` 用规范化字符串 startswith；生成预算依据 gold token 数 | 使用冻结别名和完整答案评分，统一输出预算；模糊语义项单独裁决 |

Pitfalls 的 G2A 已有正确方向的改进：显式 base_fact_id、fold、同关系/不同 component donor、统一 resid_pre 约定。因此最省事的路径是**在现有 exact-HF runner 上移植 PNT 的必要测量与干预模块**，用 PNT 项目作参考与方法核对；不必把所有数据重排成 demo 的十个目录，也不必重复搭一个完整实验框架。

## 5. 应保留、简化和暂缓的步骤

| 步骤 | 评审 | 处理 |
|---|---|---|
| 同模型 checkpoint/tokenizer/template/scorer | 必须保留 | 行为、归因、修复绑定一致；跨硬件以规定容差和结论稳定性核验，不能把所有浮点字节完全一致当成普适要求 |
| Original/Neutral/Targeted + Natural | 必须保留 | 对当前主目标，option-free 三臂是主实验，MCQ 为挖掘和配对补充 |
| 独立审核最终输入、来源、别名和事实真值 | 必须保留 | 小机制 cohort 做充分证据核查；全池保留证据等级与分层审计，不强迫所有条目达到同一成本 |
| fact/component split，训练 donor 与评估隔离 | 必须保留 | 文本翻译接触与目标行为调参接触分开记账；不因增加一种语言重做所有 component |
| 四代理模型作为 exact-HF 前必经关卡 | 对主闭环可省 | 已有结果保留为跨模型迁移附录；未来直接以主 exact-HF 挖掘，代理 panel 按研究问题选做 |
| 每事实两攻击 variant | 先导有价值，扩量可简化 | 保留旧160；扩展按现协议一个指定攻击目标加一个 non-target foil |
| 多轮重复审核直到全接受 | 不合理 | 有界生成/审核，修改字段定向复核；未改内容绑定复用；reject/defer 保留，禁止“重抽到通过” |
| 全池精细关系分类与全量最终上下文 | 过早 | 只对选中机制/行为 cohort 补齐；一般行为不要求所有 relation 都有正式细分类 |
| 扩展所有语言后才尝试修复 | 阻塞核心反馈 | 五语资源已完成先保留，优先一个模型/一个目标语言闭环，再扩同事实语言比较 |
| 多份手工状态文档、层层包装 manifest | 可合并 | 一个活动任务入口、一份状态摘要、少量阶段 manifest；机器检查保留，旧版本只读归档 |
| 4折×35层×7尺度×多控制 | 合理但昂贵 | 先在 Development 做有界可行性与层定位；预声明粗到细搜索。旧冻结网格改动必须发布新版本，不能事后缩到表现最好的层 |
| 极小 holdout 上强显著性/细关系结论 | 不合理 | 32条保留为先导确认，报告区间；正式扩展规模按精度/效应预算决定 |

随机控制不能因省预算而删除；对选中的候选设置做足对应控制，避免在每个无效配置上机械展开全部成本。OOF 用于选择配置的 Development 分数仍含选择偏差，最终效果须看独立集。

## 6. 对 task 制定的具体调整建议

已有任务设计的优点是：同一 exact-HF、三臂、按事实计数、隔离 holdout、OOF、不用 proxy 阳性替代 target 标签。五语协议已经写明只对选择 cohort 补齐最终输入、不机械重算 split，这些优化应保留。

当前需要改的不是增加更多 gate，而是明确每个 task 回答哪个研究问题。建议下一版活动任务压缩为：

| 顺序 | Task / 要回答的问题 | 交付物与完成标准 |
|---|---|---|
| T0 | 固定有效缺陷标准、恢复可执行材料 | 主终点=开放生成；扰动家族不限并保留标签；补齐原冻结包与模型/render/activation 索引，验 hash；明确现160为先导 |
| T1 | MCQ 能否找到开放召回缺陷？ | 完整96的 option-free 三臂与 MCQ交叉表；分别算生成迁移/打分迁移；全部失败、阴性与对照保留 |
| T2 | 扰动改变了哪个可检验阶段？ | 答案文本层轨迹；回忆/输出/上下文采信候选解释；小范围因果 patch/ablation，明确未解释项 |
| T3 | PNT 干预能否修复？ | 同一个HF/同输入，no-op/random/task/difference/combined；source-fold隔离；生成正确率、指定错误率及净伤害 |
| T4 | 冻结规则能否独立确认？ | 保持旧32+32；选层/scale只能用Development。变更方法后旧validation若已看过必须降为开发证据 |
| T5 | 结论能否扩展到语言/关系/模型？ | 从已完成五语共同池选择预定义cohort；完整最终输入与open行为；第二模型/未见语言或关系按明确泛化问题追加 |

T1 如果没有开放迁移，应结束“这些样本可用于开放缺陷修复”的推断。可以保留 MCQ 论文结果，并发布新 Development 攻击协议继续寻找 open 缺陷。目标不是保证出现正结果。

现有96的桥接也用于判断是否保留MCQ前筛：后续新候选可直接以开放回答反馈优化扰动，若MCQ没有带来足够的调用成本节约或额外发现率，不将它设为每个候选的必经步骤。

规模建议是先完成现有96的桥接，随后从共同池按 relation/source/component 预先抽取一个有预算上限的 Development 子集（例如几百条作为工程量级，不是功效保证）。把 Natural 放在昂贵扰动生成和深度审计之前用于新 Development 的筛选；筛选分母与规则公开。验证集在冻结后一次评测完整选定名单，可预声明分层，不能依据结果反复换样本。

当前 `task_plan.md` 的部分旧 checklist 仍表示 exact-HF 未开始，`tasks/README.md` 顶部仍叙述9月30日翻译续跑，最新 manifest 却已结束。这是状态维护问题。建议活动状态从产物生成摘要，历史事件与下一步明确分栏，避免旧文字再次驱动错误任务。

## 7. 最终如何利用数据完成整体分析

### 数据分层与关联

不要仅输出一个“badcases.json”。保留完整候选集合和对照，在同一数据包内用四张逻辑表关联：

1. **fact**：base_fact_id、source、canonical fact、aliases、relation、component、split、证据层级。
2. **stimulus**：fact_id、language、format(MCQ/open)、manipulation_family、arm、variant、distractor、上下文、最终 render/hash。
3. **behavior**：stimulus_id、精确模型身份、实际生成、语义正确性、指定错误命中、sequence scores、运行状态、判分版本。
4. **mechanism/intervention**：stimulus_id、layer/hook/position、测量、vector-source partition、方法/scale、干预后生成、恢复与新增伤害。

同一事实跨语言、arm、格式、模型均可关联；失败样本与抗扰动对照也随包发布。模型缺陷是 `(fact, stimulus, model, scoring_version)` 的属性，不应永久写成事实本身有缺陷。

### 五组主要分析

| 研究问题 | 主分析 | 必要限制 |
|---|---|---|
| 数据构建质量 | 来源→抽取→审核→翻译→最终输入→可评估漏斗；分source/relation/lang | 翻译接受数不是缺陷数量；共同集是筛选后的总体 |
| 挖掘器价值 | MCQ阳性/阴性 × open定向缺陷，含跨模型复现 | 只测MCQ阳性无法估计漏检；阳性选择过的比例不作总体发生率 |
| 跨语言扰动效应 | 同事实 Δ_lang = accuracy(Targeted)−accuracy(Neutral)，再对比 Δ_lang−Δ_EN；同时报指定错误率 | 完整开放事实集合为主，严格EN保留子集单列；不同语言单语集合不直接混比分数 |
| 机制归因 | 同事实Neutral→Targeted答案轨迹，抗扰动对照，因果patch与对照 | 不能把按final错误分组产生的logit差异当作新的因果证据 |
| 修复与代价 | 固定规则在独立集的生成正确率变化、错误→正确恢复率、正确→错误伤害率；task/diff/combined消融 | open缺陷条件结果与全部冻结事实的净收益同时报告；不只对已知失败触发修复 |

误差分析至少区分回忆候选失败、输出语言候选失败、上下文采信、MCQ-only、基础知识/评分不明，允许“暂未解释”。对英目标语同形答案不强行解读语言转换。

事实是计数单位，相关事实通过 leakage component 进行块重抽样；语言和variant不能膨胀样本量。先定一个主终点，其他指标作解释或敏感性分析。小关系格只描述；多个语言/多个预设假设的检验采用适当多重比较控制。

目前32条独立确认集适合先导，不能可靠支撑十余关系的细分结论。以独立二项样本p≈0.5的近似计算，n=32的95%误差范围约±17个百分点；相关性或paired endpoint会改变精度，不能直接据此给最终样本量。正式规模应基于主终点、配对不一致率、component结构与希望达到的区间宽度设计。

### 使用现有五语共同池

逐条重算共同集为3,326条，development=1,983、validation=673、sealed=670。可以保留完整共同集作为资源，但先选机制cohort，不必全部跑每层干预。其较大分组中包括 `definition_term_direction_unresolved` 327条，因此“关系分组可用”不等于正式机制关系已明确。

可先选答案较短、方向明确且关系有足够 donor 的子集检验方法；保留长答案/同形答案/未解析关系作为独立报告层，避免把最容易分析的子集包装成全部QA分布。是否跨关系、跨语言迁移，必须分别冻结 vector-source 和 evaluation 的边界。

## 8. Claim audit

| Claim | Verdict | 已有证据 | 缺失证据 / 建议表述 |
|---|---|---|---|
| 已形成五语审核数据资源 | keep | 4,513行与3,326共同接受逐条可核对 | 明确是翻译接受资源，尚非最终开放缺陷集 |
| 已发现Qwen3的MCQ扰动缺陷 | keep | 960条结果、192variant标签、hash一致 | 声明模型、扰动家族和Development范围 |
| 已构建开放事实召回扰动缺陷集 | revise | 自然baseline与option-free文本规格 | 尚需option-free三臂实际生成与标签 |
| 已用PNT完成机制归因 | weaken | MCQ decision-position attribution | 当前只支持MCQ内部描述；还缺开放答案与因果检验 |
| 已实现PNT修复 | remove | 当前只有离线合同 | 需要向量生成、干预、独立确认和伤害控制结果 |
| PNT的英文回忆→目标语输出机制适用于当前模型 | revise | 原论文提供方法与假设 | 对Qwen3/当前任务需检验，不能先验宣布成立 |

## 9. 关键证据入口

- [160权威任务](/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/tasks/task_160_mcq_pnt_protocol.md)
- [五语任务与最终输入定义](/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/tasks/task_multilingual_expansion_protocol.md)
- [G2A准备代码](/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/scripts/prepare_qwen3_g2a_offline_contract.py)
- [160扰动家族配置](/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/configs/factual_perturbation_zh_dual_distractor_proxy_v8.json)
- [PNT干预demo](/Users/qiqing/learn_resource/mutil-language/paths_not_taken/demo_intervention_performance.py)
- [PNT ICL构造](/Users/qiqing/learn_resource/mutil-language/paths_not_taken/prepare_ICL_fv.py)
- [PNT loader](/Users/qiqing/learn_resource/mutil-language/paths_not_taken/demo_load_datasets_model.py)
- [本次计数与hash核验](/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/analysis_output/pitfalls_pnt_review_20261009/evidence.json)
- [PNT原论文](https://aclanthology.org/2025.emnlp-main.762/)；方法细节另参照[arXiv v2](https://arxiv.org/html/2505.20546v2)，两版本不视为完全相同。

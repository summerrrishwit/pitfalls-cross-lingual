## 2026-11-03 最新离线核验

中文生产与翻译审核已于北京时间 2026-10-27 结束；接受候选 4,807 条，非去重事实数。
活动 worktree 762b 的独有代码、测试、协议、终局与审核账本已按原字节同步到主目录。
历史 160 条修复停于 Validation G3 未通过，Sealed 未开启。
数据清理分类、快照哈希验证与提交准备见 `analysis_output/status_git_audit_20261103/`。
本轮没有删除实验数据、重跑模型或修改历史冻结绑定。

## 2026-10-24 当前数据建设方案

以[中文候选流水线](/Users/qiqing/.codex/worktrees/762b/pitfalls-cross-lingual/tasks/task_zh_candidate_pipeline.md)及 analysis_output/active_data_pipeline.json 为准：Qwen规范化英文并翻译中文，Luna只审核翻译，Sol只裁决翻译疑难项。旧关系逐条复审方案、专用运行器、配置、测试及派生数据已删除；仅保留删除清单和历史曝光元数据。原始抽取账本、历史译文、160实验及当前7,250条来源输入保留。中文通过不自动满足正式实验/PNT冻结。

# Notes: Full Qwen Relation Normalization

## Current Inventory
- Input: `data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/relation_inventory.json`
- Extracted records: 2,744
- Relation signatures: 2,425
- Singleton signatures: 2,315
- Signatures with count >= 2: 110
- Signatures with count >= 5: 19

## Contract
- A signature is keyed by normalized `relation_raw`, `subject_type`, and `answer_type`.
- Taxonomy relations define ID, definition, normalized subject type, normalized answer type, direction, and examples.
- Mapping decisions must be explicit and versioned.
- Unknown or unsafe signatures remain `ambiguous` or `out_of_taxonomy` with null normalized fields.
- Applying identical artifacts must be deterministic and preserve source/extraction provenance.

## Pending Findings
- Relation family profile
  - `term for definition` covers 428 records but means description-to-term.
  - `definition` and `meaning` mean term-to-definition.
  - `classification`, `is`, `name`, and other broad labels are overloaded and require type-aware or unresolved handling.
  - Initial lexical family coverage: definition 512 records; classification 248; location 110; cause/effect 110; measurement 96; function/purpose 56; requirement 51; part/whole 44.
- Candidate taxonomy size and coverage
  - Candidate relations: 33
  - Mapped signatures: 792 / 2,425 (32.66%)
  - Mapped records: 1,083 / 2,744 (39.47%)
  - Ambiguous: 221 signatures / 247 records
  - Out of taxonomy: 1,412 signatures / 1,414 records
- Ambiguous and out-of-taxonomy review queues
  - Persisted in `relation_review_queue_candidate_v1.json`, ordered by status and record support.
- Determinism and provenance checks
  - 13,625 input and output records with identical candidate order.
  - Every original input field is unchanged in normalized output.
  - All 10,881 non-extracted records are `not_applicable`.
  - Two in-memory applications are byte-equivalent after canonical JSON serialization.
  - Normalized canonical SHA-256: `1837e166041ea4254096fafdf064a14b52d20adf01ceec9da3cb033cf6da4030`.
# 2026-10-20: bounded open-answer discovery update

- User requests an open-generation defect workflow using source wrong options as perturbation
  material, with limited Development feedback and independent-fact yield per request budget.
- Current analyzer derives open labels independently but indexes MCQ labels unconditionally,
  requires 960 inputs, and emits facts only for exactly two variants. These are migration
  assumptions, not requirements of open discovery.
- Preserve existing uncommitted remote-experiment code and all frozen evidence. New-candidate
  policies must not reinterpret an exposed Validation set as untouched.
- Current bridge evidence: 17 open-directed facts, including 5 MCQ-negative facts; this supports
  removing MCQ-positive admission. Historical counts remain linked to their original runs.
- Implemented optional-MCQ/variable-candidate analysis and a bounded-ledger efficiency checker;
  synchronized the active protocol and five-language input contract. 27 relevant tests passed,
  and the legacy 960-input metrics reproduced in a temporary directory. New inference was not run.
- Verification found a trailing blank line added to this notes file; removed it before delivery.

## 2026-10-24 用户授权删除旧关系复审中间数据

已确认当前中文流程不依赖旧关系审核结果，随后永久删除旧relation_expansion归档的10个payload目录（13061文件，约2.00GB）。原归档不再提供旧提案、Luna/Sol关系裁决或旧接受候选集；仅保留哈希清单、删除凭据和不含语义正文的历史曝光元数据。原始账本、当前7250输入/中文流程、旧译文/160实验、.env及通用脚本保留；不得恢复旧worker。详情见analysis_output/relation_intermediate_cleanup_20261024_v1/cleanup_report.md。此前“可恢复归档”描述仅反映删除前历史。

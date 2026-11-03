# 2026-10-20 开放缺陷生成流程更新

新候选使用：来源QA及错误选项素材 → 无选项Original → 一个Targeted与匹配Neutral →
实际开放生成判定 → 独立复核与复现 → 机制归因和修复。

## 已完成

- 新活动入口：`tasks/task_open_factual_discovery_protocol.md`；根README、tasks入口、任务计划和五语协议已同步。
- Development限制首轮一个候选、首轮汇总后最多一次修订，成功或到限即停，不递归堆叠context。
- `configs/open_factual_discovery_v1.json`定义范围、候选、策略版本、请求和token上限；不启动推理。
- `scripts/analyze_qwen3_option_free.py`不再强制MCQ标签、960输入、96事实或双扰动；
  单扰动、MCQ阴性和完全没有MCQ标签都可产生开放标签，缺失事实保留为未知。
- `scripts/summarize_open_discovery.py`离线检查候选/请求账本，按独立复现缺陷事实计算产出，
  计入技术失败、重试、批请求、审核和复跑，报告对照稳定性、排除率、扰动家族及可用成本。

## 验证

- 开放分析相关测试11项通过，预算与效率账本测试10项通过，既有开放评分测试6项通过。
- 在临时目录重放已有960条/96事实结果，fact_counts、MCQ交叉表、generation_metrics、
  语义缺失数、输入数、variant数与完成状态均与既有汇总一致；未覆盖原始产物。
- 本轮没有新模型调用、向量或干预执行。

## 实际执行前仍需

为新cohort配置实际目标模型运行器、逐请求账本和发请求前的预算拦截。现有旧160专用runner
仍绑定其旧输入与模型合同，不将离线分析支持误写为新候选生成已执行。
已暴露的Validation不得重新作为未见验证；既有冻结产物与Scoring版本按原版保留。

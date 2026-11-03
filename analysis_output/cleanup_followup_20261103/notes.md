# 依据与边界

- 当前活动协议：tasks/task_zh_candidate_pipeline.md、tasks/task_open_factual_discovery_protocol.md。
- 当前终局：analysis_output/zh_normalize_translate_20261023_v1/final_summary.json。
- CodeGraph 确认导出器读取 input、batches、review_packets 及其绑定；relation_inventory 为早期盘点文件，不是该导出器运行输入。
- 删除未来主流程不消费的文件，仍可能影响历史复现或哈希清单；主流程依赖与历史证据保留分开判断。
- 4807 接受记录里有2003关系ID，136个ID在接受池内有多个完整定义，涉及425记录。
- NFKC/大小写/空白归一 subject、原 relation_id、归一 answer 得到4785键，22重复余量；不等于已确认可删除22事实。
- 1848关系ID只有一条记录，31个ID至少5条；关系映射未闭合，此分布不能直接作为最终PNT样本量。
- 当前中文轮次已终结。既有1848技术失败、263校验失败、151规范化defer、3规范化reject和6翻译defer不是自动重试待办。
- Git 暂存区保持原样；本目录为新增未暂存分析文档。

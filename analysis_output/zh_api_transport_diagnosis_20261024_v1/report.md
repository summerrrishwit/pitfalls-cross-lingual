# Qwen接口修复与验证

验证时间：2026-10-24T13:39:43.873179+00:00。生产 worker PID 32407，使用单一 SSE 流式传输入口，保持原模型、路由、密钥、8并发、每请求2条、8192输出上限，以及全部提示词、schema和审核标准。

此前非流式生产请求持续返回502/503/504及零用量网关错误。DNS、TLS和鉴权验证正常。相同合成输入、相同8192输出上限的对照中，非流式两次返回503，流式两次完成。该证据支持采用流式传输避开当前失败路径；客户端不能确认服务商内部根因。

最终版本的单请求1/1、8并发8/8均通过，36/36合成案例匹配，模型标识正确且usage非零。59项回归测试通过。完整结束标记、稳定模型标识、非零usage、完整JSON及原校验全部满足后才入账；断流、截断和网关错误不会成为候选。

旧worker 99215通过本次修复专属STOP标记排空请求并正常退出，未发送强制信号。确认退出、锁和哈希后启动新worker；旧检查点逐字节不变，全部既有review/dispatch、失败尝试、恢复归档、exposure/attempt前缀均已验证。旧失败仍保留，原有一次有上限tail补试策略继续生效。

截至导出快照，规范化 1292/7250（修复前1266），中文翻译 1234/7250（修复前1212）。新流式生产成功请求分阶段计数：{'normalize': 13, 'translate': 11}；失败校验：{'answer_leaked_in_prompt': 1, 'RemoteProtocolError': 6, 'InternalServerError': 2}，均真实保留。中文审核1212条，完整状态分母合计7250。新就绪包仍由独立Luna逐条审核，只有真实翻译defer进入Sol。packet0061本次完成12条审核，最终11接受、1拒绝。

原定时任务已原位更新为v4流式策略，原30分钟调度、活动状态和目标聊天保持不变。后续自动探测使用绑定的流式probe，并核验其代码及流式组装器哈希。

本报告仅确认接口修复后生产已恢复增长，不代表7250条全量完成，也不保证上游永不再出现临时故障。中文候选尚非正式实验样本，formal/PNT门禁仍为false。

证据：verification.json、connectivity_report.json、paired_probe_report.json、stream_final_probe/report.json、activation_result.json、automation_update_verified.json；运行目录service_health.json及recovery_history/stream_transport_20261024_v1。

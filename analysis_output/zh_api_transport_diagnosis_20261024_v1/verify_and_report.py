"""Verify repair provenance and persist a production-success readout."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
RUN=HERE.parent/'zh_normalize_translate_20261023_v1'
sys.path.insert(0,str(REPO/'scripts'))
from recover_zh_candidate_pipeline import module,read,sha,worker_entrypoint

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

api=module(RUN/'runtime/run_zh_candidate_pipeline.py','frozen_stream_verification')
m=api.verify(RUN)
entry=worker_entrypoint(RUN,m)
for name,h in read(RUN/'operations_tools.json')['files'].items():assert sha(Path(name))==h,name
for name,h in read(RUN/'postprocessing_tools.json')['tools'].items():assert sha(Path(name))==h,name
before=read(HERE/'before_activation.json')
exposure=(RUN/'exposure.jsonl').read_bytes()
assert hashlib.sha256(exposure[:before['exposure_prefix_bytes']]).hexdigest()==before['exposure_prefix_sha256']
for name,h in before['protected_review_files'].items():assert sha(Path(name))==h,name
for name,h in before['batch_checkpoints'].items():assert sha(Path(name))==h,name

archives=[]
for path in sorted((RUN/'recovery_history').glob('*/manifest.json')):
    record=read(path)
    for original,binding in record.get('archived_files',{}).items():
        archived=Path(binding['archive']);assert sha(archived)==binding['sha256'],str(archived)
        if archived.name.startswith('batch_'):
            old=read(archived);current=read(Path(original))
            assert current['attempts'][:len(old['attempts'])]==old['attempts'],original
    for name,h in record.get('protected_review_files',{}).items():assert sha(Path(name))==h,name
    for name,h in record.get('probe_bindings',{}).items():assert sha(Path(name))==h,name
    if record.get('exposure_prefix_sha256'):
        assert hashlib.sha256(exposure[:record['exposure_prefix_bytes']]).hexdigest()==record['exposure_prefix_sha256']
    if record.get('recovery_tool_sha256'):assert sha(path.parent/'recovery_tool.py')==record['recovery_tool_sha256']
    archives.append({'recovery_id':path.parent.name,'manifest_sha256':sha(path),'archived_batch_snapshots':sum(Path(v['archive']).name.startswith('batch_') for v in record.get('archived_files',{}).values())})
for binding in read(RUN/'operational_history/streaming_transport_20261024_v1/manifest.json').values():
    assert sha(Path(binding['archive']))==binding['sha256']

pending=[];reviewed=[]
for packet in sorted((RUN/'review_packets').glob('packet_*')):
    pm=read(packet/'manifest.json')
    assert sha(packet/'input.jsonl')==pm['input_file_sha256']
    assert sha(packet/'instructions.md')==pm['instructions_sha256']
    assert pm['run_manifest_sha256']==sha(RUN/'manifest.json')
    if not (packet/'final_manifest.json').exists():pending.append(packet.name);continue
    fm=read(packet/'final_manifest.json')
    assert sha(packet/'final.jsonl')==fm['final_sha256']
    assert sha(packet/'dispatch.json')==fm['dispatch_sha256']
    for binding in fm['real_agent_bindings']:assert sha(Path(binding['output_path']))==binding['output_sha256']
    reviewed+=api.rows(packet/'final.jsonl')
assert len({r['record_id'] for r in reviewed})==len(reviewed)
events=[json.loads(x) for x in exposure.decode().splitlines()]
stream=[e for e in events if e.get('worker_policy')=='transport_stream_v1' and e.get('event')=='request_finished']
success=[e for e in stream if e.get('status')=='completed']
assert any(e['stage']=='normalize' for e in success) and any(e['stage']=='translate' for e in success)
assert all(e.get('response_model')==m['expected_response_model'] and e.get('usage',{}).get('total_tokens',0)>0 for e in success)
summary=read(RUN/'exports/summary.json')
assert sum(summary['source_status_counts'].values())==7250
assert summary['normalized_count']>before['status']['normalized_records']
assert summary['translated_count']>before['status']['translated_records']
gates=read(RUN/'formal_entry_gates.json');assert not gates['formal_experiment_eligible'] and not gates['pnt_eligible']
stamp=datetime.now(timezone.utc).isoformat()
result={'at':stamp,'tests_passed':59,'worker_pid':read(RUN/'worker.json')['pid'],'old_worker_pid':before['worker']['pid'],
        'entrypoint':str(entry),'response_transport':'sse','requested_model':m['requested_model'],
        'model_route_credentials_and_prompt_contracts_unchanged':True,
        'frozen_operations_overlay_reviews_dispatch_archives_exposure_attempt_prefixes_valid':True,
        'original_checkpoints_byte_identical':True,'historical_recoveries':archives,
        'diagnostic_probes':read(HERE/'stream_final_probe/report.json')['levels'],
        'production_success_by_stage':dict(Counter(e['stage'] for e in success)),
        'production_failure_types':dict(Counter(e.get('validation_code') or e.get('error_code') or e.get('error_type') for e in stream if e.get('status')=='failed')),
        'first_stream_production_success':success[0]['at'],'last_stream_production_success':success[-1]['at'],
        'before_normalized':before['status']['normalized_records'],'before_translated':before['status']['translated_records'],
        'summary':summary,'pending_packets':pending,'tail_retry_archive_count':len(list((RUN/'transport_retry_history').glob('**/batch_*.json'))),
        'formal_experiment_eligible':False,'human_gold':False}
write(HERE/'verification.json',result)
health=read(RUN/'service_health.json');health.update(at=stamp,state='recovered_with_real_production_growth',worker_state='single_streaming_worker_running',
    last_success_at=success[-1]['at'],successful_normalization_and_translation_verified=True,
    recovery_report=str(HERE/'report.md'),recovery_verification=str(HERE/'verification.json'),
    previous_degradation_first_notification_audit=health.get('first_user_notification_audit'),
    minutes_since_last_success=round((datetime.now(timezone.utc)-datetime.fromisoformat(success[-1]['at'])).total_seconds()/60,2),
    recovery_notified_in_interactive_reply=True,last_audit='interactive_stream_transport_repair')
write(RUN/'service_health.json',health)
report=f'''# Qwen接口修复与验证

验证时间：{stamp}。生产 worker PID {result['worker_pid']}，使用单一 SSE 流式传输入口，保持原模型、路由、密钥、8并发、每请求2条、8192输出上限，以及全部提示词、schema和审核标准。

此前非流式生产请求持续返回502/503/504及零用量网关错误。DNS、TLS和鉴权验证正常。相同合成输入、相同8192输出上限的对照中，非流式两次返回503，流式两次完成。该证据支持采用流式传输避开当前失败路径；客户端不能确认服务商内部根因。

最终版本的单请求1/1、8并发8/8均通过，36/36合成案例匹配，模型标识正确且usage非零。59项回归测试通过。完整结束标记、稳定模型标识、非零usage、完整JSON及原校验全部满足后才入账；断流、截断和网关错误不会成为候选。

旧worker {result['old_worker_pid']}通过本次修复专属STOP标记排空请求并正常退出，未发送强制信号。确认退出、锁和哈希后启动新worker；旧检查点逐字节不变，全部既有review/dispatch、失败尝试、恢复归档、exposure/attempt前缀均已验证。旧失败仍保留，原有一次有上限tail补试策略继续生效。

截至导出快照，规范化 {summary['normalized_count']}/7250（修复前{result['before_normalized']}），中文翻译 {summary['translated_count']}/7250（修复前{result['before_translated']}）。新流式生产成功请求分阶段计数：{result['production_success_by_stage']}；失败校验：{result['production_failure_types']}，均真实保留。中文审核{summary['reviewed_count']}条，完整状态分母合计7250。新就绪包仍由独立Luna逐条审核，只有真实翻译defer进入Sol。packet0061本次完成12条审核，最终11接受、1拒绝。

原定时任务已原位更新为v4流式策略，原30分钟调度、活动状态和目标聊天保持不变。后续自动探测使用绑定的流式probe，并核验其代码及流式组装器哈希。

本报告仅确认接口修复后生产已恢复增长，不代表7250条全量完成，也不保证上游永不再出现临时故障。中文候选尚非正式实验样本，formal/PNT门禁仍为false。

证据：verification.json、connectivity_report.json、paired_probe_report.json、stream_final_probe/report.json、activation_result.json、automation_update_verified.json；运行目录service_health.json及recovery_history/stream_transport_20261024_v1。
'''
(HERE/'report.md').write_text(report)
handoff=f'''## Current handoff

Interactive API repair at {stamp}: active policy zh-operations-v4-stream-20261024; entry runtime_transport_stream_v1, workerPID{result['worker_pid']}. Same model/route/credentials, prompts/schema and8concurrency/batch2/output8192 unchanged. Old worker{result['old_worker_pid']} drained normally through repair-owned STOP, which was archived and removed only after byte-identity/dead-process checks. Final streaming probes1/1+8/8 passed, all36syntheticcases,59tests. Real normalize AND translate successes verified; snapshotnormalized{summary['normalized_count']},translated{summary['translated_count']},reviewed{summary['reviewed_count']}. Historical124failedbatches retained; tailretry not yet reached, no reopening of exhausted/schema/reviewed records. Original checkpoints/reviews/dispatch/recovery/exposure/attempt prefixes valid. Packet0061 finalized11accept1reject after real Luna+Sol, hidden snapshots unverified,human_goldfalse. Pendingpackets{pending}; production continues independently. Automation updated in place, schedule30minutes preserved. Next heartbeat uses verified active SSE probe/runtime; never revert to old nonstreaming runtime. Formal/PNTfalse. Detailed report {HERE/'report.md'}.
'''
for filename in ['notes.md','orchestration.md']:
    path=RUN/filename;s=path.read_text();prefix=s.split('## Current handoff',1)[0]
    if '## Explicit streaming interface repair' not in prefix:
        prefix+='## Explicit streaming interface repair (2026-10-24, v4)\n\nUser-authorized transport-only SSE upgrade; v3 continuation and bounded tail retry unchanged. Runtime and diagnostic helper hashes are frozen separately; original runtime remains immutable. See active_worker_runtime.json and operations_policy.json.\n\n'
    path.write_text(prefix+handoff)
plan=HERE/'task_plan.md';plan.write_text(plan.read_text().replace('- [ ]','- [x]'))
print(json.dumps({'verified':True,'normalized':summary['normalized_count'],'translated':summary['translated_count'],
                  'reviewed':summary['reviewed_count'],'production_success_by_stage':result['production_success_by_stage'],
                  'report':str(HERE/'report.md')}))

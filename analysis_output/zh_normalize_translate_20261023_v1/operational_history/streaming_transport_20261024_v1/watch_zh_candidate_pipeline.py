#!/usr/bin/env python3
"""Inspect the frozen pipeline and perform bounded, authorized technical recovery."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from recover_zh_candidate_pipeline import module, read, recover, sha, technical, worker_entrypoint, health


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def worker(root: Path, manifest: dict) -> dict:
    info = read(root / 'worker.json') if (root / 'worker.json').exists() else {}
    pid = info.get('pid')
    if not isinstance(pid, int) or pid <= 1:
        return {'alive': False, 'owned': False, 'pid': None}
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    if result.returncode == 1 and not result.stdout.strip():
        return {'alive': False, 'owned': False, 'pid': pid}
    if result.returncode != 0:
        raise ValueError('process_identity_unavailable')
    command = shlex.split(result.stdout.strip())
    expected = [manifest['python'], str(worker_entrypoint(root, manifest)),
                'work', '--run-dir', str(root)]
    return {'alive': True, 'owned': command == expected, 'pid': pid}


def recovery_budget(events: list[dict], now: float, policy: dict) -> tuple[bool, str]:
    attempts = [e for e in events if e.get('event') == 'recovery_attempt_started' and now - timestamp(e['at']) < 86400]
    if any(e.get('manual_block') for e in events):
        return False, 'manual_block_recorded'
    if len(attempts) >= policy['max_attempts_per_24h']:
        return False, 'daily_recovery_budget_exhausted'
    if attempts:
        delay = policy['backoff_minutes'][min(len(attempts) - 1, len(policy['backoff_minutes']) - 1)] * 60
        if now - timestamp(attempts[-1]['at']) < delay:
            return False, 'recovery_cooldown'
    return True, 'recovery_budget_available'


def classify(status: dict, fatal: dict | None, error: dict | None, process: dict,
             last_activity: float, previous: dict, fingerprint: str, now: float, policy: dict,
             retryable: int, stopped: bool = False) -> dict:
    result = {'at': datetime.fromtimestamp(now, timezone.utc).isoformat(), 'fingerprint': fingerprint,
              'pid': process['pid'], 'eligible_repair': False, 'last_activity_age_seconds': max(0, now - last_activity)}
    if stopped:
        return {**result, 'state': 'user_stopped'}
    if process['alive'] and not process['owned']:
        return {**result, 'state': 'manual_process_identity_mismatch'}
    if fatal and fatal.get('error_code') not in {'gateway_zero_usage_error_response', 'systematic_gateway_errors'}:
        return {**result, 'state': 'manual_fatal_error'}
    if error and error.get('error_type') not in {'APIConnectionError', 'APITimeoutError', 'OSError'}:
        return {**result, 'state': 'manual_worker_error'}
    if not process['alive']:
        if fatal:
            return {**result, 'state': 'recoverable_gateway', 'eligible_repair': True}
        if status.get('state') == 'qwen_finished_reviews_pending':
            return {**result, 'state': 'recoverable_technical_batches' if retryable else 'qwen_finished_reviews_pending',
                    'eligible_repair': bool(retryable)}
        if process['pid'] and status.get('state') in {'running_qwen', 'explicit_recovery_launching', 'stopped_after_stall'}:
            return {**result, 'state': 'recoverable_worker_exit', 'eligible_repair': True}
        return {**result, 'state': 'manual_unexplained_worker_state'}
    same = previous.get('fingerprint') == fingerprint and previous.get('pid') == process['pid']
    since = previous.get('unchanged_since', previous.get('at', result['at'])) if same else result['at']
    result['unchanged_since'] = since
    if now - last_activity < policy['stall_minutes'] * 60:
        return {**result, 'state': 'healthy_active'}
    confirmed = same and now - timestamp(since) >= policy['confirmation_minutes'] * 60
    return {**result, 'state': 'stalled_confirmed' if confirmed else 'stall_suspected', 'eligible_repair': confirmed}


def inspect(root: Path, policy: dict) -> tuple[dict, object, dict]:
    api = module(root / 'runtime/run_zh_candidate_pipeline.py', 'frozen_zh_watch_api')
    manifest = api.verify(root)
    worker_entrypoint(root, manifest)  # Verify any separately bound runtime overlay even when the worker is dead.
    status = read(root / 'status.json')
    fatal = read(root / 'fatal.json') if (root / 'fatal.json').exists() else None
    error = read(root / 'worker_error.json') if (root / 'worker_error.json').exists() else None
    observations = root / 'automatic_recovery/observation.json'
    previous = read(observations) if observations.exists() else {}
    events = api.rows(root / 'exposure.jsonl')
    activity = [timestamp(e['at']) for e in events[-100:] if e.get('at')]
    info = read(root / 'worker.json') if (root / 'worker.json').exists() else {}
    activity += [timestamp(info[k]) for k in ['started_at', 'launched_at'] if info.get(k)]
    last_activity = max(activity) if activity else timestamp(status['updated_at'])
    state = {k: status.get(k) for k in ['normalized_records', 'translated_records', 'batch_status_counts']}
    fingerprint = api.digest([state, len(events), events[-1] if events else None])
    retryable = sum(b.get('status') == 'failed' and not b.get('recovery_history') and bool(b.get('attempts'))
                    and technical(b['attempts'][-1]) for b in (read(p) for p in (root / 'batches').glob('batch_*.json')))
    result = classify(status, fatal, error, worker(root, manifest), last_activity, previous,
                      fingerprint, time.time(), policy, retryable, (root / 'STOP').exists())
    result.update(retryable_technical_batches=retryable, policy_sha256=sha(root / 'operations_policy.json'))
    api.atomic(observations, result)
    return result, api, manifest


def run(root: Path, apply: bool = False) -> dict:
    root = root.resolve()
    bindings = root / 'operations_tools.json'
    if bindings.exists():
        if any(sha(Path(p)) != expected for p, expected in read(bindings)['files'].items()):
            raise ValueError('operations_tool_binding_changed')
    policy = read(root / 'operations_policy.json')['automatic_recovery']
    folder = root / 'automatic_recovery'
    folder.mkdir(exist_ok=True)
    with (folder / 'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, api, manifest = inspect(root, policy)
        ledger = folder / 'events.jsonl'
        events = api.rows(ledger)
        allowed, reason = recovery_budget(events, time.time(), policy)
        result.update(recovery_budget_available=allowed, budget_reason=reason, repair_applied=False)
        if not apply or not result['eligible_repair'] or not allowed:
            return result
        attempt_id = 'auto_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        attempt = folder / 'attempts' / attempt_id
        attempt.mkdir(parents=True)
        def event(value: dict) -> None:
            with ledger.open('a') as f:
                f.write(json.dumps({'at': api.now(), 'attempt_id': attempt_id, **value}) + '\n')
        event({'event': 'recovery_attempt_started', 'trigger': result['state']})
        api.atomic(attempt / 'before.json', result)
        if result['state'] == 'stalled_confirmed':
            # Re-check both ownership and freshness immediately before a graceful stop.
            current = worker(root, manifest)
            if not current['alive'] or not current['owned'] or current['pid'] != result['pid']:
                event({'event': 'recovery_attempt_finished', 'result': 'process_changed_no_signal'})
                return {**result, 'attempt_id': attempt_id, 'action': 'process_changed_no_signal'}
            latest = api.rows(root / 'exposure.jsonl')
            if latest and time.time() - timestamp(latest[-1]['at']) < policy['stall_minutes'] * 60:
                event({'event': 'recovery_attempt_finished', 'result': 'activity_resumed_no_signal'})
                return {**result, 'attempt_id': attempt_id, 'action': 'activity_resumed_no_signal'}
            os.kill(current['pid'], signal.SIGTERM)
            for _ in range(100):
                if not worker(root, manifest)['alive']:
                    break
                time.sleep(.2)
            else:
                event({'event': 'recovery_attempt_finished', 'result': 'graceful_stop_failed', 'manual_block': True})
                return {**result, 'attempt_id': attempt_id, 'action': 'manual_graceful_stop_failed'}
            api.progress(root, 'stopped_after_stall')
        reports = []
        for concurrency in [1, manifest['concurrency']]:
            dest = attempt / f'probe_c{concurrency}'
            command = [manifest['python'], str(Path(manifest['runtime']) / 'probe_relation_expansion_api.py'),
                       '--env-file', manifest['env_file'], '--output-dir', str(dest), '--levels', str(concurrency),
                       '--waves', '1', '--timeout', str(policy['probe_timeout_seconds']), '--max-output-tokens', '8192']
            try:
                probe = subprocess.run(command, capture_output=True, text=True, timeout=240)
            except subprocess.TimeoutExpired:
                event({'event': 'recovery_attempt_finished', 'result': 'probe_process_timeout'})
                return {**result, 'attempt_id': attempt_id, 'action': 'probe_failed_cooldown'}
            diagnostic = api.rows(dest / f'requests_c{concurrency}.jsonl')
            manual = any(x.get('http_status') in {401, 403, 404}
                         or x.get('status') == 'completed' and x.get('response_model') != manifest['expected_response_model']
                         for x in diagnostic)
            partial_ok = False
            overlay = root / 'active_worker_runtime.json'
            if overlay.exists() and concurrency > 1 and not manual:
                route_api = module(root / 'runtime/probe_relation_expansion_api.py', 'watch_overlay_route')
                _, base, model = route_api.load_route(Path(manifest['env_file']))
                try:
                    health(reports + [dest / 'report.json'],
                           {**manifest, 'transport_probe_min_success_ratio': read(overlay)['transport_probe_min_success_ratio']},
                           base, model)
                    partial_ok = True
                except (ValueError, KeyError, OSError):
                    partial_ok = False
            if (probe.returncode != 0 and not partial_ok) or manual:
                event({'event': 'recovery_attempt_finished', 'result': 'probe_failed', 'manual_block': manual,
                       'concurrency': concurrency, 'returncode': probe.returncode})
                return {**result, 'attempt_id': attempt_id, 'action': 'manual_probe_failure' if manual else 'probe_failed_cooldown'}
            reports.append(dest / 'report.json')
        try:
            recovered = recover(root, attempt_id, reports, apply=True, limit=policy['max_failed_batches_per_attempt'])
        except Exception as error:
            event({'event': 'recovery_attempt_finished', 'result': 'recovery_refused', 'error_type': type(error).__name__})
            raise
        event({'event': 'recovery_attempt_finished', 'result': 'worker_launched', 'worker_pid': recovered['worker_pid']})
        output = {**result, 'attempt_id': attempt_id, 'repair_applied': True, 'worker_pid': recovered['worker_pid']}
        api.atomic(attempt / 'result.json', output)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.run_dir, args.repair), ensure_ascii=False))
    except Exception as error:
        code = str(error) if re.fullmatch('[a-z_]+', str(error)) else None
        print(json.dumps({'state': 'manual_attention', 'error_type': type(error).__name__, 'error_code': code}))
        raise SystemExit(1)

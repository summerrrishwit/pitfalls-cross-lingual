#!/usr/bin/env python3
"""Explicit, health-gated recovery around an unchanged frozen Qwen runtime."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def worker_entrypoint(root: Path, manifest: dict) -> Path:
    authority_path = root / 'active_worker_runtime.json'
    original = Path(manifest['runtime']) / 'run_zh_candidate_pipeline.py'
    if not authority_path.exists():
        return original
    authority = read(authority_path)
    if authority['parent_manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('worker_parent_manifest_changed')
    entry = Path(authority['entrypoint']).resolve()
    if root.resolve() not in entry.parents or str(entry) not in authority['bindings']:
        raise ValueError('invalid_worker_runtime_path')
    for path, expected in authority['bindings'].items():
        if sha(Path(path)) != expected:
            raise ValueError('worker_runtime_binding_changed')
    return entry


def health(reports: list[Path], manifest: dict, base: str, model: str) -> dict:
    """Require current single-request and frozen-concurrency synthetic evidence."""
    expected_route = hashlib.sha256((base.rstrip('/') + '\n' + model).encode()).hexdigest()
    evidence = {}
    levels = set()
    for report_path in reports:
        report = read(report_path)
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(report['created_at'])).total_seconds()
        if not -60 <= age <= 3600:
            raise ValueError('probe_evidence_stale')
        if (report.get('route_sha256') != expected_route
                or report.get('requested_model') != model
                or report.get('probe_source_sha256') != sha(Path(manifest['runtime']) / 'probe_relation_expansion_api.py')
                or report.get('observed_response_model') != manifest['expected_response_model']
                or report.get('input_policy') != 'synthetic_static_cases_only'
                or report.get('retries') != 0):
            raise ValueError('probe_identity_or_route_mismatch')
        evidence[str(report_path)] = sha(report_path)
        for level in report['levels']:
            c = level['concurrency']
            request_path = report_path.parent / f'requests_c{c}.jsonl'
            requests = [json.loads(line) for line in request_path.read_text().splitlines()]
            partial = manifest.get('transport_probe_min_success_ratio')
            if partial is not None and not .75 <= partial <= 1:
                raise ValueError('invalid_partial_probe_threshold')
            completed = [r for r in requests if r.get('status') == 'completed']
            partial_ok = (partial is not None and c > 1 and len(completed) >= c * partial
                          and all((r.get('status') == 'completed' or technical(r))
                                  and r.get('http_status') not in {401, 403, 404} for r in requests))
            checked = completed if partial_ok else requests
            if ((not level.get('bounded_probe_passed') or level.get('failed') != 0) and not partial_ok
                    or len(requests) != level['requests']
                    or level['completed'] != len(checked)
                    or level['peak_in_flight'] != c or len(requests) < c
                    or len({r['request_index'] for r in requests}) != len(requests)
                    or any(r.get('status') != 'completed'
                           or r.get('response_model') != manifest['expected_response_model']
                           or r.get('usage', {}).get('total_tokens', 0) <= 0
                           or r.get('semantic_case_matches') != 4 for r in checked)):
                raise ValueError('probe_request_evidence_failed')
            levels.add(c)
            evidence[str(request_path)] = sha(request_path)
    if not {1, manifest['concurrency']} <= levels:
        raise ValueError('single_and_concurrent_probe_required')
    return evidence


def technical(attempt: dict) -> bool:
    status = attempt.get('http_status')
    return (attempt.get('error_code') == 'gateway_zero_usage_error_response'
            or attempt.get('error_type') in {'APIConnectionError', 'APITimeoutError', 'InternalServerError', 'RateLimitError'}
            or isinstance(status, int) and (status == 429 or status >= 500))


def inventory(root: Path, api: Any, manifest: dict, limit: int) -> tuple[list, dict]:
    source = api.rows(root / 'input.jsonl')
    if len(source) != manifest['source_count']:
        raise ValueError('source_count_changed')
    claimed = set()
    protected = {}
    for packet in sorted((root / 'review_packets').glob('packet_*')):
        pm = read(packet / 'manifest.json')
        if (sha(packet / 'input.jsonl') != pm['input_file_sha256']
                or sha(packet / 'instructions.md') != pm['instructions_sha256']
                or sha(root / 'manifest.json') != pm['run_manifest_sha256']):
            raise ValueError('review_packet_binding_changed')
        for row in api.rows(packet / 'input.jsonl'):
            if row['record_id'] in claimed:
                raise ValueError('duplicate_packet_record')
            claimed.add(row['record_id'])
        for p in packet.iterdir():
            if p.is_file():
                protected[str(p)] = sha(p)
        if (packet / 'final_manifest.json').exists():
            fm = read(packet / 'final_manifest.json')
            if sha(packet / 'final.jsonl') != fm['final_sha256'] or sha(packet / 'dispatch.json') != fm['dispatch_sha256']:
                raise ValueError('final_review_binding_changed')
            for binding in fm['real_agent_bindings']:
                p = Path(binding['output_path'])
                if sha(p) != binding['output_sha256']:
                    raise ValueError('review_output_changed')
                protected[str(p)] = sha(p)
    selected = []
    for path in sorted((root / 'batches').glob('batch_*.json')):
        value = read(path)
        if value.get('status') != 'failed':
            continue
        attempts = value.get('attempts', [])
        if not attempts or not technical(attempts[-1]):
            continue  # Semantic/schema-only failures are not silently reopened.
        if value.get('recovery_history'):
            continue  # Exhausted batches remain failed; do not block untouched work.
        start = int(path.stem.split('_')[1]) * manifest['batch_size']
        batch = source[start:start + manifest['batch_size']]
        if value['input_sha256'] != api.digest(batch) or value.get('translated'):
            raise ValueError('failed_checkpoint_binding_changed')
        if claimed & {r['record_id'] for r in batch}:
            raise ValueError('reviewed_record_cannot_be_requeued')
        selected.append((path, value))
    if len(selected) > limit:
        raise ValueError('failed_batch_recovery_limit_exceeded')
    return selected, protected


def recover(root: Path, recovery_id: str, reports: list[Path], apply: bool = False,
            limit: int = 32, spawn: Any = subprocess.Popen) -> dict:
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', recovery_id) or not 0 <= limit <= 128:
        raise ValueError('invalid_recovery_scope')
    root = root.resolve()
    if (root / 'operations_tools.json').exists():
        if any(sha(Path(p)) != expected for p, expected in read(root / 'operations_tools.json')['files'].items()):
            raise ValueError('operations_tool_binding_changed')
    api = module(root / 'runtime/run_zh_candidate_pipeline.py', 'frozen_zh_recovery_api')
    manifest = api.verify(root)
    entrypoint = worker_entrypoint(root, manifest)
    overlay = read(root / 'active_worker_runtime.json') if (root / 'active_worker_runtime.json').exists() else {}
    if overlay:
        manifest = {**manifest, 'transport_probe_min_success_ratio': overlay['transport_probe_min_success_ratio']}
    route_api = module(root / 'runtime/probe_relation_expansion_api.py', 'frozen_route_recovery_api')
    _, base, model = route_api.load_route(Path(manifest['env_file']))
    if api.digest([base.rstrip('/'), model]) != manifest['route_sha256']:
        raise ValueError('configured_route_changed')
    evidence = health([p.resolve() for p in reports], manifest, base, model)
    target = root / 'recovery_history' / recovery_id
    with ExitStack() as stack:
        launch_lock = stack.enter_context((root / 'launch.lock').open('a+'))
        worker_lock = stack.enter_context((root / 'worker.lock').open('a+'))
        for lock in [launch_lock, worker_lock]:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'worker.json').exists():
            try:
                os.kill(read(root / 'worker.json')['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('worker_already_exists')
        if (root / 'STOP').exists():
            raise ValueError('user_stop_must_not_be_overridden')
        fatal = read(root / 'fatal.json') if (root / 'fatal.json').exists() else None
        if fatal is not None and fatal.get('error_code') not in {'gateway_zero_usage_error_response', 'systematic_gateway_errors'}:
            raise ValueError('only_gateway_failure_recovery_allowed')
        if fatal is None:
            state = read(root / 'status.json').get('state')
            error = read(root / 'worker_error.json') if (root / 'worker_error.json').exists() else None
            if (not (root / 'worker.json').exists()
                    or state not in {'running_qwen', 'explicit_recovery_launching', 'qwen_finished_reviews_pending', 'stopped_after_stall'}
                    or error and error.get('error_type') not in {'APIConnectionError', 'APITimeoutError', 'OSError'}):
                raise ValueError('unexplained_exit_requires_manual_review')
        if target.exists():
            raise ValueError('recovery_id_already_exists')
        selected, protected = inventory(root, api, manifest, limit)
        if fatal is None and state == 'qwen_finished_reviews_pending' and not selected:
            raise ValueError('no_technical_work_to_retry')
        result = {'recovery_id': recovery_id, 'at': api.now(), 'state': 'validated_plan',
                  'retry_batches': [p.name for p, _ in selected], 'probe_bindings': evidence,
                  'frozen_manifest_sha256': sha(root / 'manifest.json'), 'protected_review_files': protected,
                  'frozen_bindings': manifest['bindings'], 'human_gold': False,
                  'formal_experiment_eligible': False, 'automatic_semantic_repairs': 0,
                  'recovery_trigger': fatal.get('error_code') if fatal else 'verified_worker_exit',
                  'excluded_failed_batches': [p.name for p in (root / 'batches').glob('batch_*.json')
                                             if read(p).get('status') == 'failed' and p not in {x[0] for x in selected}]}
        if not apply:
            return result
        target.mkdir(parents=True)
        shutil.copyfile(Path(__file__), target / 'recovery_tool.py')
        result['recovery_tool_sha256'] = sha(target / 'recovery_tool.py')
        originals = [root / n for n in ['fatal.json', 'status.json', 'worker.json', 'worker_error.json'] if (root / n).exists()]
        originals += [p for p, _ in selected]
        result['archived_files'] = {}
        for p in originals:
            dest = target / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            if sha(dest) != sha(p):
                raise ValueError('recovery_archive_copy_mismatch')
            result['archived_files'][str(p)] = {'archive': str(dest), 'sha256': sha(dest)}
        result['exposure_prefix_bytes'] = (root / 'exposure.jsonl').stat().st_size if (root / 'exposure.jsonl').exists() else 0
        result['exposure_prefix_sha256'] = sha(root / 'exposure.jsonl') if (root / 'exposure.jsonl').exists() else None
        result['state'] = 'archived_before_requeue'
        api.atomic(target / 'manifest.json', result)
        for path, value in ([] if overlay.get('defer_technical_retries_to_tail') else selected):
            value['recovery_history'] = [{'recovery_id': recovery_id, 'previous_status': 'failed',
                                          **result['archived_files'][str(path)]}]
            value['status'] = 'normalized' if value.get('normalized') else 'recovery_pending'
            api.atomic(path, value)  # Existing attempts and normalized fields stay intact.
        for p, expected in protected.items():
            if sha(Path(p)) != expected:
                raise ValueError('protected_review_changed_during_recovery')
        api.verify(root)
        # Remove only active stop markers after their byte-identical archived evidence exists.
        if (root / 'fatal.json').exists():
            (root / 'fatal.json').unlink()
        if (root / 'worker_error.json').exists():
            (root / 'worker_error.json').unlink()
        with (root / 'exposure.jsonl').open('a') as f:
            f.write(json.dumps({'at': api.now(), 'event': 'explicit_gateway_recovery',
                                'recovery_id': recovery_id, 'retry_batches': result['retry_batches'],
                                'target_behavior_exposure': False}) + '\n')
        api.progress(root, 'explicit_recovery_launching')
        command = [manifest['python'], str(entrypoint),
                   'work', '--run-dir', str(root)]
        worker_lock.close()  # Child acquires the original runtime lock; launch lock stays held.
        try:
            with (root / 'worker.log').open('a') as log:
                child = spawn(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=root)
        except Exception:
            if (target / 'fatal.json').exists():
                shutil.copyfile(target / 'fatal.json', root / 'fatal.json')
            else:
                api.atomic(root / 'worker_error.json', {'at': api.now(), 'error_type': 'recovery_launch_failed',
                                                       'recovery_id': recovery_id})
            result['state'] = 'launch_failed_fatal_restored' if fatal else 'launch_failed_guard_recorded'
            api.atomic(target / 'manifest.json', result)
            api.progress(root, 'blocked_api')
            raise
        api.atomic(root / 'worker.json', {'pid': child.pid, 'launched_at': api.now(), 'recovery_id': recovery_id})
        result.update(state='launched', worker_pid=child.pid, command=command)
        api.atomic(target / 'manifest.json', result)
        api.atomic(root / 'active_recovery.json', {'recovery_id': recovery_id, 'manifest': str(target / 'manifest.json'),
                                                 'manifest_sha256': sha(target / 'manifest.json'), 'worker_pid': child.pid})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--recovery-id', required=True)
    parser.add_argument('--probe-report', type=Path, action='append', required=True)
    parser.add_argument('--max-failed-batches', type=int, default=32)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = recover(args.run_dir, args.recovery_id, args.probe_report, args.apply, args.max_failed_batches)
    print(json.dumps({k: v for k, v in result.items() if k not in {'protected_review_files', 'frozen_bindings'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()

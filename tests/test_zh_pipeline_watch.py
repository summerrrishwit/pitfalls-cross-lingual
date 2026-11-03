import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_zh_candidate_pipeline as w
from test_zh_candidate_recovery import run

POLICY = {'stall_minutes': 20, 'confirmation_minutes': 10, 'max_attempts_per_24h': 3,
          'backoff_minutes': [30, 60], 'probe_timeout_seconds': 90, 'max_failed_batches_per_attempt': 32}
NOW = 1_800_000_000.0


def iso(at):
    return datetime.fromtimestamp(at, timezone.utc).isoformat()


def classify(**kw):
    args = dict(status={'state': 'running_qwen'}, fatal=None, error=None,
                process={'pid': 42, 'alive': True, 'owned': True}, last_activity=NOW - 3000,
                previous={}, fingerprint='same', now=NOW, policy=POLICY, retryable=0)
    args.update(kw)
    return w.classify(**args)


def test_slow_request_or_first_stale_observation_does_not_restart():
    assert classify(last_activity=NOW - 299)['state'] == 'healthy_active'
    first = classify()
    assert first['state'] == 'stall_suspected' and not first['eligible_repair']
    second = classify(previous={**first, 'at': iso(NOW - 900), 'unchanged_since': iso(NOW - 900)})
    assert second['state'] == 'stalled_confirmed' and second['eligible_repair']


@pytest.mark.parametrize('change', [{'fingerprint': 'new'}, {'process': {'pid': 43, 'alive': True, 'owned': True}}])
def test_progress_or_new_worker_resets_stall_confirmation(change):
    old = {'fingerprint': 'same', 'pid': 42, 'unchanged_since': iso(NOW - 900)}
    assert classify(previous=old, **change)['state'] == 'stall_suspected'


def test_auth_identity_unknown_error_and_user_stop_never_auto_repair():
    dead = {'pid': 42, 'alive': False, 'owned': False}
    for fatal in [{'error_code': 'api_access_or_model_unavailable'}, {'error_code': 'response_model_identity_mismatch'}]:
        assert not classify(process=dead, fatal=fatal)['eligible_repair']
    assert not classify(error={'error_type': 'ValueError'})['eligible_repair']
    assert not classify(stopped=True)['eligible_repair']
    assert classify(process={'pid': 42, 'alive': True, 'owned': False})['state'] == 'manual_process_identity_mismatch'


def test_gateway_exit_and_only_remaining_technical_work_are_repairable():
    dead = {'pid': 42, 'alive': False, 'owned': False}
    assert classify(process=dead, fatal={'error_code': 'systematic_gateway_errors'})['eligible_repair']
    assert classify(process=dead)['eligible_repair']
    done = {'state': 'qwen_finished_reviews_pending'}
    assert not classify(status=done, process=dead)['eligible_repair']
    assert classify(status=done, process=dead, retryable=1)['eligible_repair']


def test_recovery_budget_backoff_daily_cap_and_manual_latch():
    attempt = lambda ago: {'event': 'recovery_attempt_started', 'at': iso(NOW - ago)}
    assert w.recovery_budget([], NOW, POLICY)[0]
    assert w.recovery_budget([attempt(300)], NOW, POLICY)[1] == 'recovery_cooldown'
    assert w.recovery_budget([attempt(1801)], NOW, POLICY)[0]
    assert w.recovery_budget([attempt(7200), attempt(3000)], NOW, POLICY)[1] == 'recovery_cooldown'
    assert w.recovery_budget([attempt(9000), attempt(8000), attempt(7000)], NOW, POLICY)[1] == 'daily_recovery_budget_exhausted'
    assert w.recovery_budget([attempt(90000)], NOW, POLICY)[0]
    assert not w.recovery_budget([{'manual_block': True}], NOW, POLICY)[0]


def test_healthy_inspection_with_repair_flag_never_probes_or_restarts(run, monkeypatch):
    root, _, api = run
    (root / 'fatal.json').unlink()
    api.atomic(root / 'status.json', {'state': 'running_qwen', 'updated_at': datetime.now(timezone.utc).isoformat()})
    api.atomic(root / 'operations_policy.json', {'automatic_recovery': POLICY})
    monkeypatch.setattr(w, 'worker', lambda *_: {'pid': 42, 'alive': True, 'owned': True})
    def forbidden(*args, **kwargs):
        raise AssertionError('healthy worker must never probe or restart')
    monkeypatch.setattr(w.subprocess, 'run', forbidden)
    monkeypatch.setattr(w, 'recover', forbidden)
    result = w.run(root, apply=True)
    assert result['state'] == 'healthy_active' and not result['repair_applied']
    assert not (root / 'automatic_recovery/events.jsonl').exists()


def test_operations_hash_drift_blocks_before_any_process_or_api_action(run):
    root, _, api = run
    api.atomic(root / 'operations_tools.json', {'files': {str(root / 'input.jsonl'): 'wrong_hash'}})
    with pytest.raises(ValueError, match='operations_tool_binding_changed'):
        w.run(root, apply=True)
    assert not (root / 'automatic_recovery').exists()


def test_auto_recovery_runs_two_bounded_probes_archives_and_obeys_cooldown(run, monkeypatch):
    root, reports, api = run
    api.atomic(root / 'status.json', {'state': 'blocked_api', 'updated_at': datetime.now(timezone.utc).isoformat()})
    api.atomic(root / 'operations_policy.json', {'automatic_recovery': POLICY})
    monkeypatch.setattr(w, 'worker', lambda *_: {'pid': 42, 'alive': False, 'owned': False})
    calls = []
    def fake_probe(command, **kwargs):
        concurrency = int(command[command.index('--levels') + 1])
        calls.append(concurrency)
        dest = Path(command[command.index('--output-dir') + 1])
        shutil.copytree(reports[0 if concurrency == 1 else 1].parent, dest)
        return SimpleNamespace(returncode=0)
    original = w.recover
    monkeypatch.setattr(w.subprocess, 'run', fake_probe)
    monkeypatch.setattr(w, 'recover', lambda *args, **kwargs: original(
        *args, **kwargs, spawn=lambda *a, **k: SimpleNamespace(pid=12345)))
    result = w.run(root, apply=True)
    assert calls == [1, 8] and result['repair_applied']
    assert (root / 'recovery_history' / result['attempt_id'] / 'fatal.json').exists()
    assert not (root / 'fatal.json').exists()
    repeated = w.run(root, apply=True)
    assert repeated['budget_reason'] == 'recovery_cooldown'
    assert calls == [1, 8]
    assert len([e for e in api.rows(root / 'automatic_recovery/events.jsonl')
                if e['event'] == 'recovery_attempt_started']) == 1


def test_auth_probe_failure_latches_manual_block_without_production_restart(run, monkeypatch):
    root, _, api = run
    api.atomic(root / 'status.json', {'state': 'blocked_api', 'updated_at': datetime.now(timezone.utc).isoformat()})
    api.atomic(root / 'operations_policy.json', {'automatic_recovery': POLICY})
    monkeypatch.setattr(w, 'worker', lambda *_: {'pid': 42, 'alive': False, 'owned': False})
    calls = []
    def unauthorized(command, **kwargs):
        calls.append(command)
        dest = Path(command[command.index('--output-dir') + 1]); dest.mkdir()
        api.write_rows(dest / 'requests_c1.jsonl', [{'status': 'failed', 'http_status': 401}])
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(w.subprocess, 'run', unauthorized)
    monkeypatch.setattr(w, 'recover', lambda *a, **k: pytest.fail('must not restart on authentication failure'))
    result = w.run(root, apply=True)
    assert result['action'] == 'manual_probe_failure'
    assert (root / 'fatal.json').exists()
    assert not (root / 'recovery_history').exists()
    again = w.run(root, apply=True)
    assert again['budget_reason'] == 'manual_block_recorded'
    assert len(calls) == 1

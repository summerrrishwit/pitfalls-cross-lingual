import importlib.util
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('zh_recovery', ROOT / 'scripts/recover_zh_candidate_pipeline.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    runtime = root / 'runtime'
    runtime.mkdir(parents=True)
    shutil.copyfile(ROOT / 'scripts/run_zh_candidate_pipeline.py', runtime / 'run_zh_candidate_pipeline.py')
    (runtime / 'probe_relation_expansion_api.py').write_text(
        "def load_route(path): return ('not-a-real-secret', 'route', 'qwen-3.8-max')\n")
    api = recovery.module(runtime / 'run_zh_candidate_pipeline.py', 'fixture_frozen_api')
    source = [{'record_id': f'source:{i}', 'answer': 'Mira', 'source_answer': 'Mira',
               'source_question': 'Who wrote Sample Book?'} for i in range(2)]
    api.write_rows(root / 'input.jsonl', source)
    dump(root / 'relation_catalog.json', {'relations': []})
    bindings = {str(p): recovery.sha(p) for p in [root / 'input.jsonl', root / 'relation_catalog.json', *runtime.glob('*.py')]}
    manifest = {'bindings': bindings, 'source_count': 2, 'runtime': str(runtime), 'env_file': 'unused',
                'expected_response_model': 'Qwen-3.8-Max', 'requested_model': 'qwen-3.8-max',
                'route_sha256': api.digest(['route', 'qwen-3.8-max']), 'concurrency': 8,
                'batch_size': 1, 'review_packet_size': 20, 'execution_mode': 'pipeline', 'python': 'python'}
    dump(root / 'manifest.json', manifest)
    for i in range(2):
        api.atomic(root / f'batches/batch_{i:05d}.json', {
            'status': 'failed', 'input_sha256': api.digest([source[i]]),
            'normalized': [] if i == 0 else [{'record_id': 'source:1', 'decision': 'defer', 'reason': 'insufficient evidence'}],
            'translated': [], 'attempts': [{'stage': 'normalize', 'status': 'failed', 'error_type': 'APIConnectionError'}]})
    (root / 'review_packets').mkdir()
    dump(root / 'fatal.json', {'error_code': 'gateway_zero_usage_error_response', 'usage': {'total_tokens': 0}})
    dump(root / 'worker.json', {'pid': 987654321})
    dump(root / 'status.json', {'state': 'blocked_api'})
    (root / 'exposure.jsonl').write_text('{"event":"historical_attempt"}\n')
    dump(root / 'formal_entry_gates.json', {'formal_experiment_eligible': False})
    reports = []
    for c in [1, 8]:
        folder = tmp_path / f'probe{c}'
        folder.mkdir()
        report = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'route_sha256': __import__('hashlib').sha256(b'route\nqwen-3.8-max').hexdigest(),
                  'requested_model': 'qwen-3.8-max', 'probe_source_sha256': recovery.sha(runtime / 'probe_relation_expansion_api.py'),
                  'observed_response_model': 'Qwen-3.8-Max', 'input_policy': 'synthetic_static_cases_only', 'retries': 0,
                  'levels': [{'concurrency': c, 'requests': c, 'completed': c, 'failed': 0,
                              'peak_in_flight': c, 'bounded_probe_passed': True}]}
        dump(folder / 'report.json', report)
        api.write_rows(folder / f'requests_c{c}.jsonl', [
            {'request_index': i, 'status': 'completed', 'response_model': 'Qwen-3.8-Max',
             'usage': {'total_tokens': 10}, 'semantic_case_matches': 4} for i in range(c)])
        reports.append(folder / 'report.json')
    def dead(_pid, _signal):
        raise ProcessLookupError
    monkeypatch.setattr(recovery.os, 'kill', dead)
    return root, reports, api


def test_explicit_recovery_preserves_failures_attempts_exposure_and_frozen_artifacts(run):
    root, reports, api = run
    before = {str(p): p.read_bytes() for p in [root / 'fatal.json', root / 'exposure.jsonl',
              root / 'manifest.json', root / 'input.jsonl', root / 'runtime/run_zh_candidate_pipeline.py',
              root / 'formal_entry_gates.json', *sorted((root / 'batches').glob('*.json'))]}
    calls = []
    def spawn(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=12345)
    result = recovery.recover(root, 'one_epoch', reports, apply=True, spawn=spawn)
    assert len(calls) == 1 and calls[0][1] == str(root / 'runtime/run_zh_candidate_pipeline.py')
    assert result['state'] == 'launched' and result['worker_pid'] == 12345
    history = root / 'recovery_history/one_epoch'
    assert not (root / 'fatal.json').exists()
    assert (history / 'fatal.json').read_bytes() == before[str(root / 'fatal.json')]
    for p in sorted((root / 'batches').glob('*.json')):
        original = json.loads(before[str(p)])
        assert (history / p.relative_to(root)).read_bytes() == before[str(p)]
        current = api.read(p)
        assert current['attempts'] == original['attempts']
        assert current['normalized'] == original['normalized']
        assert current['recovery_history'][0]['sha256'] == recovery.sha(history / p.relative_to(root))
    assert api.read(root / 'batches/batch_00000.json')['status'] == 'recovery_pending'
    assert api.read(root / 'batches/batch_00001.json')['status'] == 'normalized'
    assert (root / 'exposure.jsonl').read_bytes().startswith(before[str(root / 'exposure.jsonl')])
    for n in ['manifest.json', 'input.jsonl', 'runtime/run_zh_candidate_pipeline.py', 'formal_entry_gates.json']:
        assert (root / n).read_bytes() == before[str(root / n)]


@pytest.mark.parametrize('bad', ['identity', 'zero_tokens', 'stale', 'missing_concurrency'])
def test_health_failure_does_not_unblock_or_modify_checkpoints(run, bad):
    root, reports, api = run
    if bad == 'identity':
        report = recovery.read(reports[0]); report['observed_response_model'] = 'wrong'; dump(reports[0], report)
    elif bad == 'stale':
        report = recovery.read(reports[0]); report['created_at'] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(); dump(reports[0], report)
    elif bad == 'zero_tokens':
        p = reports[0].parent / 'requests_c1.jsonl'; rows = api.rows(p); rows[0]['usage']['total_tokens'] = 0; api.write_rows(p, rows)
    else:
        reports = reports[:1]
    before = (root / 'batches/batch_00000.json').read_bytes()
    with pytest.raises(ValueError):
        recovery.recover(root, 'must_not_apply', reports, apply=True)
    assert (root / 'fatal.json').exists()
    assert (root / 'batches/batch_00000.json').read_bytes() == before
    assert not (root / 'recovery_history').exists()


def test_live_worker_and_frozen_drift_refuse_recovery(run, monkeypatch):
    root, reports, _ = run
    monkeypatch.setattr(recovery.os, 'kill', lambda *args: None)
    with pytest.raises(ValueError, match='worker_already_exists'):
        recovery.recover(root, 'duplicate', reports, apply=True)
    assert (root / 'fatal.json').exists()
    (root / 'input.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='frozen_binding_changed'):
        recovery.recover(root, 'drift', reports, apply=True)


def test_semantic_failure_not_requeued_and_retry_limit_enforced(run):
    root, reports, api = run
    with pytest.raises(ValueError, match='limit_exceeded'):
        recovery.recover(root, 'limit', reports, limit=1)
    p = root / 'batches/batch_00000.json'; b = api.read(p)
    b['attempts'][-1] = {'error_type': 'ValueError', 'validation_code': 'known_relation_spec_mismatch'}
    api.atomic(p, b)
    plan = recovery.recover(root, 'only_technical', reports)
    assert plan['retry_batches'] == ['batch_00001.json']
    assert api.read(p)['status'] == 'failed'


def test_launch_failure_restores_stop_guard_and_keeps_failure_archive(run):
    root, reports, api = run
    def broken_spawn(*args, **kwargs):
        raise OSError('simulated spawn failure')
    with pytest.raises(OSError):
        recovery.recover(root, 'launch_failure', reports, apply=True, spawn=broken_spawn)
    assert (root / 'fatal.json').exists()
    assert api.read(root / 'status.json')['state'] == 'blocked_api'
    assert api.read(root / 'recovery_history/launch_failure/manifest.json')['state'] == 'launch_failed_fatal_restored'
    assert (root / 'recovery_history/launch_failure/batches/batch_00000.json').exists()


def test_verified_unexpected_exit_can_resume_without_fabricating_fatal(run):
    root, reports, api = run
    (root / 'fatal.json').unlink()
    api.atomic(root / 'status.json', {'state': 'running_qwen'})
    result = recovery.recover(root, 'crash_resume', reports, apply=True,
                              spawn=lambda *args, **kwargs: SimpleNamespace(pid=12345))
    assert result['recovery_trigger'] == 'verified_worker_exit'
    assert not (root / 'recovery_history/crash_resume/fatal.json').exists()
    assert (root / 'recovery_history/crash_resume/status.json').exists()


def test_exhausted_batch_remains_failed_without_blocking_untouched_batch(run):
    root, reports, api = run
    p = root / 'batches/batch_00000.json'; before = api.read(p)
    before['recovery_history'] = [{'recovery_id': 'previous_epoch'}]
    api.atomic(p, before)
    plan = recovery.recover(root, 'bounded', reports)
    assert plan['retry_batches'] == ['batch_00001.json']
    assert plan['excluded_failed_batches'] == ['batch_00000.json']
    assert api.read(p) == before


def access_stop_fixture(run):
    root,reports,api=run
    at=datetime.now(timezone.utc).isoformat()
    api.atomic(root/'fatal.json',{'error_code':'api_access_or_model_unavailable','http_status':403,'at':at})
    p=root/'batches/batch_00000.json';value=api.read(p)
    attempt={'stage':'translate','status':'failed','http_status':403,'error_type':'PermissionDeniedError',
             'record_ids':['source:0'],'attempt':1,'prompt_sha256':'original-prompt'}
    value['attempts']=[attempt];api.atomic(p,value)
    with (root/'exposure.jsonl').open('a') as f:f.write(json.dumps({**attempt,'at':at,'event':'request_finished'})+'\n')
    return root,reports,api


def test_403_remains_blocked_for_default_automatic_recovery(run):
    root,reports,api=access_stop_fixture(run)
    with pytest.raises(ValueError,match='only_gateway_failure_recovery_allowed'):
        recovery.recover(root,'no_automatic_access',reports,apply=True)
    assert (root/'fatal.json').exists() and not (root/'recovery_history').exists()


def test_explicit_access_restore_archives_and_requeues_episode_once(run):
    root,reports,api=access_stop_fixture(run);original=(root/'batches/batch_00000.json').read_bytes()
    result=recovery.recover(root,'explicit_access',reports,apply=True,allow_access_restore=True,
                            spawn=lambda *a,**k:SimpleNamespace(pid=12345))
    assert result['explicit_verified_access_restore'] is True
    assert result['access_batches_requeued_once']==['batch_00000.json']
    assert (root/'recovery_history/explicit_access/batches/batch_00000.json').read_bytes()==original
    current=api.read(root/'batches/batch_00000.json')
    assert current['attempts']==json.loads(original)['attempts'] and current['recovery_history']
    assert recovery.access_interrupted_batches(root,api,api.read(root/'manifest.json'),
        api.read(root/'recovery_history/explicit_access/fatal.json'))==[]


@pytest.mark.parametrize('status',[401,404])
def test_explicit_403_recovery_cannot_override_other_access_errors(run,status):
    root,reports,api=access_stop_fixture(run);fatal=api.read(root/'fatal.json');fatal['http_status']=status;api.atomic(root/'fatal.json',fatal)
    with pytest.raises(ValueError,match='requires_recorded_403'):
        recovery.recover(root,'wrong_access',reports,apply=True,allow_access_restore=True)
    assert (root/'fatal.json').exists()


def test_explicit_access_restore_requires_exposure_episode_evidence(run):
    root,reports,api=access_stop_fixture(run);(root/'exposure.jsonl').write_text('')
    assert recovery.access_interrupted_batches(root,api,api.read(root/'manifest.json'),api.read(root/'fatal.json'))==[]


def test_explicit_access_restore_rejects_partial_concurrency_probe(run):
    root,reports,api=access_stop_fixture(run);p=reports[1].parent/'requests_c8.jsonl';rows=api.rows(p)
    rows[-1]={'request_index':7,'status':'failed','error_type':'APIConnectionError'};api.write_rows(p,rows)
    report=api.read(reports[1]);report['levels'][0].update(completed=7,failed=1,bounded_probe_passed=False);api.atomic(reports[1],report)
    with pytest.raises(ValueError,match='probe_request_evidence_failed'):
        recovery.recover(root,'partial_access',reports,apply=True,allow_access_restore=True)
    assert (root/'fatal.json').exists() and not (root/'recovery_history').exists()


def test_unknown_program_error_cannot_be_automatically_resumed(run):
    root, reports, api = run
    (root / 'fatal.json').unlink()
    api.atomic(root / 'status.json', {'state': 'running_qwen'})
    api.atomic(root / 'worker_error.json', {'error_type': 'ValueError', 'error_code': 'unknown_bug'})
    with pytest.raises(ValueError, match='unexplained_exit'):
        recovery.recover(root, 'no_bypass', reports, apply=True)

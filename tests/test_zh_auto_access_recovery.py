import shutil
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_zh_candidate_recovery import run,recovery
from test_zh_pipeline_watch import POLICY,NOW,classify,iso
import watch_zh_candidate_pipeline as w

AUTO={**POLICY,'revalidate_unknown_403':True,'access_403_cooldown_minutes':30}
DEAD={'pid':42,'alive':False,'owned':False}


def fatal403(age=3600):
    return {'error_code':'api_access_or_model_unavailable','http_status':403,'at':iso(NOW-age)}


def test_unknown403_only_becomes_repairable_after_cooldown_and_worker_exit():
    assert classify(fatal=fatal403(100),process=DEAD,policy=AUTO)['state']=='access_403_cooldown'
    v=classify(fatal=fatal403(),process=DEAD,policy=AUTO)
    assert v['state']=='recoverable_access_403' and v['eligible_repair']
    assert classify(fatal=fatal403(),policy=AUTO)['state']=='draining_access_failure'
    assert not classify(fatal=fatal403(),process=DEAD)['eligible_repair']
    assert not classify(fatal=fatal403(),process=DEAD,policy=AUTO,stopped=True)['eligible_repair']


@pytest.mark.parametrize('detail',[
    {'provider_error_code':'invalid_api_key'}, {'provider_error_code':'model_not_authorized'},
    {'provider_error_message':'权限不足'}, {'provider_error_message':'IP不在白名单'},
    {'provider_error_code':'insufficient_quota'}, {'provider_error_message':'API key expired'},
])
def test_explicit_permanent_access_errors_do_not_auto_probe(detail):
    v=classify(fatal={**fatal403(),**detail},process=DEAD,policy=AUTO)
    assert v['state']=='manual_fatal_error' and not v['eligible_repair']


def setup403(root,api):
    api.atomic(root/'fatal.json',{'error_code':'api_access_or_model_unavailable','http_status':403,
        'at':(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()})
    api.atomic(root/'status.json',{'state':'blocked_api','updated_at':datetime.now(timezone.utc).isoformat()})
    api.atomic(root/'operations_policy.json',{'automatic_recovery':AUTO})


def test_full_probe_success_archives403_then_launches_once(run,monkeypatch):
    root,reports,api=run;setup403(root,api)
    before=(root/'fatal.json').read_bytes();calls=[]
    monkeypatch.setattr(w,'worker',lambda *_:DEAD)
    def probe(command,**kwargs):
        c=int(command[command.index('--levels')+1]);calls.append(c)
        shutil.copytree(reports[0 if c==1 else 1].parent,Path(command[command.index('--output-dir')+1]))
        return SimpleNamespace(returncode=0)
    original=w.recover
    monkeypatch.setattr(w.subprocess,'run',probe)
    monkeypatch.setattr(w,'recover',lambda *a,**k:original(*a,**k,spawn=lambda *a,**k:SimpleNamespace(pid=12345)))
    value=w.run(root,apply=True)
    assert value['repair_applied'] and calls==[1,8]
    archive=root/'recovery_history'/value['attempt_id'];assert (archive/'fatal.json').read_bytes()==before
    result=api.read(archive/'manifest.json');assert result['automatic_verified_access_restore']
    assert not result['explicit_verified_access_restore']
    assert not (root/'fatal.json').exists()
    assert w.run(root,apply=True)['budget_reason']=='recovery_cooldown' and calls==[1,8]


def test_unknown403_probe_failure_keeps_fatal_and_cools_without_permanent_latch(run,monkeypatch):
    root,_,api=run;setup403(root,api);calls=[]
    monkeypatch.setattr(w,'worker',lambda *_:DEAD)
    def refused(command,**kwargs):
        calls.append(command);dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
        api.write_rows(dest/'requests_c1.jsonl',[{'status':'failed','http_status':403}])
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(w.subprocess,'run',refused)
    monkeypatch.setattr(w,'recover',lambda *a,**k:pytest.fail('refused probe cannot launch worker'))
    value=w.run(root,apply=True);assert value['action']=='probe_failed_cooldown'
    assert (root/'fatal.json').exists() and not (root/'recovery_history').exists()
    assert not any(e.get('manual_block') for e in api.rows(root/'automatic_recovery/events.jsonl'))
    assert w.run(root,apply=True)['budget_reason']=='recovery_cooldown' and len(calls)==1


def test_auto403_recovery_requires_all_eight_successes(run,monkeypatch):
    root,reports,api=run;setup403(root,api);calls=[]
    monkeypatch.setattr(w,'worker',lambda *_:DEAD)
    def probe(command,**kwargs):
        c=int(command[command.index('--levels')+1]);calls.append(c);dest=Path(command[command.index('--output-dir')+1])
        shutil.copytree(reports[0 if c==1 else 1].parent,dest)
        if c==8:
            rows=api.rows(dest/'requests_c8.jsonl');rows[-1]={'request_index':7,'status':'failed','http_status':503};api.write_rows(dest/'requests_c8.jsonl',rows)
            v=api.read(dest/'report.json');v['levels'][0].update(completed=7,failed=1,bounded_probe_passed=False);api.atomic(dest/'report.json',v)
        return SimpleNamespace(returncode=0 if c==1 else 1)
    monkeypatch.setattr(w.subprocess,'run',probe)
    monkeypatch.setattr(w,'recover',lambda *a,**k:pytest.fail('partial403 health cannot restart'))
    value=w.run(root,apply=True);assert value['action']=='probe_failed_cooldown' and calls==[1,8]
    assert (root/'fatal.json').exists()


@pytest.mark.parametrize('change',[{'revalidate_unknown_403':False},{'access_403_cooldown_minutes':120}])
def test_direct_automatic_restore_also_enforces_authorized_policy(run,change):
    root,reports,api=run;setup403(root,api);api.atomic(root/'operations_policy.json',{'automatic_recovery':{**AUTO,**change}})
    with pytest.raises(ValueError,match='automatic_access_recovery_not_authorized'):
        recovery.recover(root,'not_allowed',reports,apply=True,automatic_access_restore=True)
    assert (root/'fatal.json').exists() and not (root/'recovery_history').exists()


def test_direct_auto_flag_cannot_bypass_guard_attempt_ledger(run):
    root,reports,api=run;setup403(root,api)
    with pytest.raises(ValueError,match='automatic_access_recovery_requires_guard_attempt'):
        recovery.recover(root,'unrecorded',reports,apply=True,automatic_access_restore=True)
    assert (root/'fatal.json').exists() and not (root/'recovery_history').exists()


@pytest.mark.parametrize('kind',['daily_cap','cooldown'])
def test_helper_rechecks_budget_even_with_a_recorded_attempt(run,kind):
    root,reports,api=run;setup403(root,api);now=datetime.now(timezone.utc)
    prior=[{'at':(now-timedelta(seconds=100+i)).isoformat(),'event':'recovery_attempt_started',
            'attempt_id':f'prior{i}','trigger':'recoverable_access_403'} for i in range(3 if kind=='daily_cap' else 1)]
    current={'at':now.isoformat(),'event':'recovery_attempt_started','attempt_id':'current','trigger':'recoverable_access_403'}
    api.write_rows(root/'automatic_recovery/events.jsonl',prior+[current])
    expected='automatic_access_recovery_requires_guard_attempt' if kind=='daily_cap' else 'automatic_access_recovery_cooldown'
    with pytest.raises(ValueError,match=expected):
        recovery.recover(root,'current',reports,apply=True,automatic_access_restore=True)
    assert (root/'fatal.json').exists() and not (root/'recovery_history').exists()

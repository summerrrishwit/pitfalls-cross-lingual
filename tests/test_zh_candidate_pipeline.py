import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('zh_pipeline',ROOT/'scripts/run_zh_candidate_pipeline.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def fixture():
    src={'record_id':'source:1','source_question':'Who wrote Sample Book?','source_answer':'Mira','subject':'Sample Book','answer':'Mira','answer_aliases_en':['Mira']}
    rel={'id':'written_work_to_author','direction':'written work -> author','definition':'Author of a written work'}
    raw={'record_id':src['record_id'],'decision':'ready','subject':'Sample Book','relation_id':rel['id'],'relation_definition':rel['definition'],'direction':rel['direction'],'answer':'Mira','canonical_fact':'Mira wrote Sample Book.','prompt_en':'Who wrote Sample Book?','evidence_quotes':['Who wrote Sample Book?','Mira'],'reason':'Source QA gives the author.'}
    return src,{'relations':[rel]},raw


def translated():
    src,cat,raw=fixture();item=p.normalize({'records':[raw]},[src],cat)[0]
    zh={'record_id':src['record_id'],'subject_zh':'示例书','prompt_zh':'谁写了示例书？','canonical_fact_zh':'米拉写了示例书。','answer_zh':'米拉','answer_aliases_zh':['米拉']}
    return p.translate({'records':[zh]},[item])[0]


@pytest.mark.parametrize('mutation,code',[
    ({'answer':'Other'},'existing_answer_changed'),
    ({'evidence_quotes':['invented evidence']},'unsupported_evidence_quote'),
    ({'direction':'author -> work'},'known_relation_spec_mismatch'),
    ({'prompt_en':'Mira wrote what?'},'answer_leaked_in_prompt')])
def test_source_and_relation_contracts(mutation,code):
    src,cat,raw=fixture();raw.update(mutation)
    with pytest.raises(ValueError,match=code):p.normalize({'records':[raw]},[src],cat)


def test_unlisted_precise_relation_is_candidate_without_independent_relation_review():
    src,cat,raw=fixture();raw.update(relation_id='book_to_named_author')
    item=p.normalize({'records':[raw]},[src],cat)[0]
    assert item['independent_relation_review'] is False
    assert item['english']['catalog_known_relation'] is False


def test_no_duplicate_source_output():
    src,cat,raw=fixture()
    with pytest.raises(ValueError,match='duplicate'):p.normalize({'records':[raw,raw]},[src],cat)


def test_alias_missing_and_answer_leak_rejected():
    item=translated();v=dict(item['translation']);v['answer_aliases_zh']=[]
    with pytest.raises(ValueError,match='alias_count'):p.translate({'records':[v]},[item])
    v=dict(item['translation']);v['prompt_zh']='米拉写了什么？'
    with pytest.raises(ValueError,match='answer_leak'):p.translate({'records':[v]},[item])


def setup_packet(tmp_path):
    root=tmp_path;item=translated();source=root/'input.jsonl';p.write_rows(source,[fixture()[0]])
    p.atomic(root/'manifest.json',{'bindings':{str(source):p.sha(source)},'source_count':1,'review_packet_size':20})
    p.atomic(root/'batches/batch_00000.json',{'status':'completed','translated':[item],'normalized':[]})
    p.pack(root,flush=True);packet=root/'review_packets/packet_0000'
    review={k:item[k] for k in ['record_id','input_sha256','english_sha256','translation_sha256']}
    review.update(verdict='accept',checks={k:True for k in p.CHECKS},reason='Chinese fields preserve English meaning and identity.',issues=[],human_gold=False)
    return root,packet,item,review


def bind_dispatch(root,packet,review,model='gpt-6-luna'):
    output=root/'luna.jsonl';p.write_rows(output,[review]);dispatch=root/'dispatch.json'
    p.atomic(dispatch,{'packet_manifest_sha256':p.sha(packet/'manifest.json'),'dispatches':[{'role':'reviewer','agent_id':'actual-test-agent-id','requested_model':model,'output_path':str(output),'output_sha256':p.sha(output)}]})
    return dispatch


def test_translation_accept_does_not_promote_relation_or_formal_experiment(tmp_path):
    root,packet,item,review=setup_packet(tmp_path);dispatch=bind_dispatch(root,packet,review)
    p.finalize(root,packet,dispatch);out=p.rows(packet/'final.jsonl')[0]
    assert out['status']=='zh_translation_accepted'
    assert out['normalization_independent_review'] is False
    assert out['formal_experiment_eligible'] is False


@pytest.mark.parametrize('mutation,code',[
    ({'english_sha256':'changed'},'review_binding_mismatch'),
    ({'checks':{k:False for k in p.CHECKS}},'review_decision_inconsistent')])
def test_bound_translation_review_cannot_be_reused_after_change(tmp_path,mutation,code):
    root,packet,item,review=setup_packet(tmp_path);review.update(mutation);dispatch=bind_dispatch(root,packet,review)
    with pytest.raises(ValueError,match=code):p.finalize(root,packet,dispatch)


def test_wrong_requested_agent_model_fails(tmp_path):
    root,packet,item,review=setup_packet(tmp_path);dispatch=bind_dispatch(root,packet,review,'gpt-5.5')
    with pytest.raises(ValueError,match='actual_agent'):p.finalize(root,packet,dispatch)


def test_translation_uncertainty_routes_only_to_sol(tmp_path):
    root,packet,item,review=setup_packet(tmp_path);review.update(verdict='defer',checks={k:False for k in p.CHECKS})
    dispatch=bind_dispatch(root,packet,review);p.finalize(root,packet,dispatch)
    assert p.rows(packet/'final.jsonl')[0]['status']=='pending_sol_translation'
    assert p.rows(packet/'sol_inputs.jsonl')[0]['review_sha256']==p.digest(review)


def test_packet_packing_does_not_duplicate_or_require_entire_queue(tmp_path):
    root,packet,item,review=setup_packet(tmp_path);p.pack(root,flush=True)
    assert len(list((root/'review_packets').glob('packet_*')))==1


@pytest.mark.parametrize('concurrency',[1,8])
def test_fatal_identity_stops_queued_work_and_records_no_candidate(tmp_path,monkeypatch,concurrency):
    import sys,threading
    from types import SimpleNamespace
    src,cat,raw=fixture();source=[{**src,'record_id':f'source:{i}'} for i in range((concurrency+2)*2)]
    p.write_rows(tmp_path/'input.jsonl',source);p.atomic(tmp_path/'relation_catalog.json',cat)
    p.atomic(tmp_path/'manifest.json',{'bindings':{str(tmp_path/'input.jsonl'):p.sha(tmp_path/'input.jsonl')},'source_count':len(source),'review_packet_size':20,'env_file':'unused','requested_model':'qwen-3.8-max','expected_response_model':'Qwen-3.8-Max','route_sha256':p.digest(['route','qwen-3.8-max']),'concurrency':concurrency,'batch_size':2,'execution_mode':'pipeline'})
    calls=[];barrier=threading.Barrier(concurrency)
    class Client:
        def __init__(self,**kwargs):self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def create(self,**kwargs):
            calls.append(kwargs);barrier.wait(timeout=5)
            return SimpleNamespace(model='wrong-model',usage=None,choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    monkeypatch.setitem(sys.modules,'openai',SimpleNamespace(OpenAI=Client))
    monkeypatch.setitem(sys.modules,'probe_relation_expansion_api',SimpleNamespace(load_route=lambda _:('private','route','qwen-3.8-max'),safe_error=lambda e:{'error_type':type(e).__name__},safe_model_label=lambda v,*_:v,gateway_error_hint=lambda *args:None))
    p.work(tmp_path)
    assert len(calls)==concurrency
    assert p.read(tmp_path/'fatal.json')['error_code']=='response_model_identity_mismatch'
    assert p.read(tmp_path/'status.json')['state']=='blocked_api'
    assert p.read(tmp_path/'status.json')['translated_records']==0
    assert not list((tmp_path/'review_packets').glob('packet_*'))


def test_export_keeps_full_denominator_and_failure_without_formal_promotion(tmp_path,monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    ex=importlib.import_module('export_zh_candidate_data')
    root,packet,item,review=setup_packet(tmp_path)
    src=fixture()[0];other={**src,'record_id':'source:2'}
    p.write_rows(root/'input.jsonl',[src,other]);m=p.read(root/'manifest.json');m.update(source_count=2,batch_size=1,bindings={str(root/'input.jsonl'):p.sha(root/'input.jsonl')});p.atomic(root/'manifest.json',m)
    # Bind updated authority before making the first actual final artifact.
    pm=p.read(packet/'manifest.json');pm['run_manifest_sha256']=p.sha(root/'manifest.json');p.atomic(packet/'manifest.json',pm)
    norm=p.normalize({'records':[fixture()[2]]},[src],fixture()[1])[0]
    p.atomic(root/'batches/batch_00000.json',{'status':'completed','normalized':[norm],'translated':[item]})
    p.atomic(root/'batches/batch_00001.json',{'status':'failed','input_sha256':p.digest([other]),'normalized':[],'translated':[]})
    dispatch=bind_dispatch(root,packet,review);p.finalize(root,packet,dispatch)
    summary=ex.export(root)
    assert summary['source_count']==2
    assert summary['source_status_counts']=={'zh_translation_accepted':1,'technical_normalization_failed':1}
    assert summary['formal_experiment_eligible'] is False
    assert len(p.rows(root/'exports/source_coverage.jsonl'))==2
    p.write_rows(root/'luna.jsonl',[{**review,'reason':'changed after finalization'}])
    with pytest.raises(ValueError,match='review_output_changed'):ex.export(root)

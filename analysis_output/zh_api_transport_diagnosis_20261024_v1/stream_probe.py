"""Test SSE transport on synthetic cases without altering production."""
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'zh_normalize_translate_20261023_v1/runtime/probe_relation_expansion_api.py'
spec=importlib.util.spec_from_file_location('frozen_probe',path)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
key,base,model=p.load_route(Path('/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/utils/.env'))
prompt=('Return only a JSON object with results: [{id, relation_leaf_id}]. '
        'Classify the stated fictional facts using only supplied text. '
        'Allowed leaf IDs: written_work_to_author, description_to_term, term_to_definition, unresolved. '
        'Do not invert a relation or invent missing dates. No external knowledge. '
        'Request identifier: c1-0. Cases:\n'
        +json.dumps([{k:v for k,v in c.items() if k!='expected'} for c in p.CASES]))
result={'at':datetime.now(timezone.utc).isoformat(),'synthetic_only':True,'human_gold':False,
        'requested_model':model,'stream':True,'max_output_tokens':8192,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
started=time.monotonic();content='';reasoning='';models=set();usage={};chunks=0
try:
 with httpx.Client(timeout=httpx.Timeout(90,connect=10)) as client:
  with client.stream('POST',base.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json={
      'model':model,'messages':[{'role':'user','content':prompt}], 'response_format':{'type':'json_object'},
      'temperature':0,'max_tokens':8192,'enable_thinking':False,'stream':True,'stream_options':{'include_usage':True}}) as response:
   result['http_status']=response.status_code
   if response.status_code!=200:
    response.read();result['error_body_sha256']=hashlib.sha256(response.content).hexdigest()
   else:
    for line in response.iter_lines():
     if time.monotonic()-started>180:raise TimeoutError('bounded_total_timeout')
     if not line.startswith('data:'):continue
     if line[5:].strip()=='[DONE]':break
     value=json.loads(line[5:]);chunks+=1
     if chunks==1:result['first_chunk_seconds']=round(time.monotonic()-started,3)
     if value.get('model'):models.add(p.safe_model_label(value['model'],key,base))
     if value.get('usage'):usage=value['usage']
     for choice in value.get('choices',[]):
      delta=choice.get('delta',{});content+=delta.get('content') or '';reasoning+=delta.get('reasoning_content') or ''
    rows=json.loads(content).get('results',[])
    by_id={x['id']:x['relation_leaf_id'] for x in rows}
    result['semantic_case_matches']=sum(by_id.get(c['id'])==c['expected'] for c in p.CASES)
    result['valid']=models=={'Qwen-3.8-Max'} and usage.get('total_tokens',0)>0 and len(rows)==4 and result['semantic_case_matches']==4
except Exception as error:result.update(p.safe_error(error))
result.update(seconds=round(time.monotonic()-started,3),response_models=sorted(models),usage=usage,chunks=chunks,content_characters=len(content),reasoning_characters=len(reasoning))
(ROOT/'stream_probe_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

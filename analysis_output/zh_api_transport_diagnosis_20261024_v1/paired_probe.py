"""One matched buffered/SSE comparison on the same route and synthetic prompt."""
import hashlib, importlib.util, json, re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import httpx

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe',ROOT.parent/'zh_normalize_translate_20261023_v1/runtime/probe_relation_expansion_api.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
key,base,model=p.load_route(Path('/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/utils/.env'))
prompt=('Return only a JSON object with results: [{id, relation_leaf_id}]. '
 'Classify the stated fictional facts using only supplied text. '
 'Allowed leaf IDs: written_work_to_author, description_to_term, term_to_definition, unresolved. '
 'Do not invert a relation or invent missing dates. No external knowledge. '
 'Request identifier: c1-0. Cases:\n'+json.dumps([{k:v for k,v in c.items() if k!='expected'} for c in p.CASES]))
def clean(value):
 text=str(value)
 for secret in [key,base,urlsplit(base).hostname]:text=text.replace(secret,'[redacted]')
 return re.sub(r'https?://\S+|sk-[\w-]+','[redacted]',text)[:1200]
def request(stream):
 started=time.monotonic();result={'stream':stream};content='';models=set();usage={}
 payload={'model':model,'messages':[{'role':'user','content':prompt}], 'temperature':0,'max_tokens':8192,
          'response_format':{'type':'json_object'},'enable_thinking':False,'stream':stream}
 if stream:payload['stream_options']={'include_usage':True}
 try:
  with httpx.Client(timeout=httpx.Timeout(90,connect=10)) as client:
   with client.stream('POST',base.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload) as response:
    result['http_status']=response.status_code
    if response.status_code!=200:
     response.read();result['error_body']=clean(response.text)
    elif stream:
     for line in response.iter_lines():
      if time.monotonic()-started>180:raise TimeoutError('bounded_total_timeout')
      if not line.startswith('data:'):continue
      if line[5:].strip()=='[DONE]':break
      chunk=json.loads(line[5:])
      if 'first_chunk_seconds' not in result:result['first_chunk_seconds']=round(time.monotonic()-started,3)
      if chunk.get('model'):models.add(p.safe_model_label(chunk['model'],key,base))
      if chunk.get('usage'):usage=chunk['usage']
      for c in chunk.get('choices',[]):content+=c.get('delta',{}).get('content') or ''
    else:
     response.read();body=response.json();models.add(p.safe_model_label(body.get('model'),key,base));usage=body.get('usage') or {}
     content=body['choices'][0]['message'].get('content') or ''
    if content:
     rows=json.loads(content).get('results',[]);by_id={x['id']:x['relation_leaf_id'] for x in rows}
     result['case_matches']=sum(by_id.get(c['id'])==c['expected'] for c in p.CASES)
     result['valid']=models=={'Qwen-3.8-Max'} and usage.get('total_tokens',0)>0 and len(rows)==4 and result['case_matches']==4
 except Exception as error:result.update(error_type=type(error).__name__,message=clean(error))
 result.update(seconds=round(time.monotonic()-started,3),models=sorted(models),usage=usage)
 return result
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(request,[False,True]))
report={'at':datetime.now(timezone.utc).isoformat(),'synthetic_only':True,'human_gold':False,'max_output_tokens':8192,
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'results':results}
(ROOT/'paired_probe_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

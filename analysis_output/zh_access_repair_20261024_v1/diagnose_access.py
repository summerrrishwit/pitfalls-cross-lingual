"""One explicit user-authorized access diagnostic; no automatic retries."""
import argparse,hashlib,importlib.util,json,re,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from dotenv import dotenv_values

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'zh_normalize_translate_20261023_v1'
parser=argparse.ArgumentParser();parser.add_argument('--structured',action='store_true');args=parser.parse_args()
m=json.loads((RUN/'manifest.json').read_text())
env=Path(m['env_file']);v=dotenv_values(env)
key,base,model=[str(v[x]).strip() for x in ['AILYUN_API_KEY','AILYUN_BASE_URL','AILYUN_MODEL']]
route=hashlib.sha256(json.dumps([base.rstrip('/'),model],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if route!=m['route_sha256']:raise ValueError('configured_route_changed')
def clean(text):
 text=str(text)
 for secret in [key,base,urlsplit(base).hostname]:
  if secret:text=text.replace(secret,'[redacted]')
 return re.sub(r'https?://\S+|sk-[\w-]+','[redacted]',text)[:2400]
report={'at':datetime.now(timezone.utc).isoformat(),'route_matches_frozen_manifest':True,'requested_model':model,
        'env_modified_at':datetime.fromtimestamp(env.stat().st_mtime,timezone.utc).isoformat(),
        'diagnostic_only':True,'retries':0,'explicit_user_authorization':'repair current HTTP403 stop',
        'credentials_endpoint_or_real_source_included':False,'human_gold':False}
payload={'model':model,'messages':[{'role':'user','content':'Return only the JSON object {"ok":true}.'}],
         'temperature':0,'max_tokens':128,'response_format':{'type':'json_object'},'enable_thinking':False,
         'stream':True,'stream_options':{'include_usage':True}}
if args.structured:
 spec=importlib.util.spec_from_file_location('frozen_access_probe',RUN/'runtime_transport_stream_v1/probe_relation_expansion_api.py')
 probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
 prompt=('Return only a JSON object with results: [{id, relation_leaf_id}]. '
         'Classify the stated fictional facts using only supplied text. '
         'Allowed leaf IDs: written_work_to_author, description_to_term, term_to_definition, unresolved. '
         'Do not invert a relation or invent missing dates. No external knowledge. '
         'Request identifier: c1-0. Cases:\n'+json.dumps([{k:v for k,v in c.items() if k!='expected'} for c in probe.CASES]))
 payload['messages'][0]['content']=prompt;payload['max_tokens']=8192
 report.update(structured_synthetic=True,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
started=time.monotonic();content='';models=set();usage={};finish=None;count=0
try:
 with httpx.Client(timeout=httpx.Timeout(90 if args.structured else 45,connect=10)) as c:
  with c.stream('POST',base.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload) as response:
   report.update(http_status=response.status_code,headers={k:clean(response.headers[k]) for k in ['server','content-type','cf-mitigated','retry-after'] if k in response.headers})
   if response.status_code!=200:
    response.read();report.update(error_body=clean(response.text),error_body_sha256=hashlib.sha256(response.content).hexdigest())
   else:
    for line in response.iter_lines():
     if time.monotonic()-started>120:raise TimeoutError('bounded_diagnostic_timeout')
     if not line.startswith('data:'):continue
     if line[5:].strip()=='[DONE]':break
     d=json.loads(line[5:]);count+=1
     if count==1:report['first_chunk_seconds']=round(time.monotonic()-started,3)
     if d.get('error'):report['stream_error']=clean(json.dumps(d['error'],ensure_ascii=False))
     if d.get('model'):models.add(clean(d['model']))
     if d.get('usage'):usage=d['usage']
     for row in d.get('choices',[]):
      content+=row.get('delta',{}).get('content') or ''
      if row.get('finish_reason'):finish=row['finish_reason']
    report.update(response_models=sorted(models),usage=usage,finish_reason=finish,content=clean(content))
    parsed=json.loads(content)
    if args.structured:
     rows=parsed.get('results',[]);by_id={x['id']:x['relation_leaf_id'] for x in rows}
     report['semantic_case_matches']=sum(by_id.get(x['id'])==x['expected'] for x in probe.CASES)
     semantic_ok=len(rows)==4 and report['semantic_case_matches']==4
    else:semantic_ok=parsed=={'ok':True}
    report['access_probe_passed']=models=={m['expected_response_model']} and usage.get('total_tokens',0)>0 and finish=='stop' and semantic_ok
except Exception as error:report.update(error_type=type(error).__name__,message=clean(error))
report['seconds']=round(time.monotonic()-started,3)
(HERE/('structured_access_diagnostic.json' if args.structured else 'access_diagnostic.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))

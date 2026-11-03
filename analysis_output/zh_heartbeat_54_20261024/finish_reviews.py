"""Bind completed real agent outputs, then run the frozen packet finalizer."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'zh_normalize_translate_20261023_v1'
parser=argparse.ArgumentParser();parser.add_argument('task');parser.add_argument('role',choices=['reviewer','adjudicator']);parser.add_argument('packets',type=int,nargs='+');args=parser.parse_args()
for number in args.packets:
    packet=ROOT/'review_packets'/f'packet_{number:04d}'
    path=packet/'dispatch.json';dispatch=json.loads(path.read_text())
    matches=[d for d in dispatch['dispatches'] if d['role']==args.role and d['agent_id']==args.task]
    assert len(matches)==1
    binding=matches[0];assert binding['status']=='running'
    output=Path(binding['output_path']);sha=hashlib.sha256(output.read_bytes()).hexdigest()
    binding.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),output_sha256=sha)
    path.write_text(json.dumps(dispatch,indent=2)+'\n')
    command=['/Users/qiqing/anaconda3/envs/clp/bin/python',str(ROOT/'runtime/run_zh_candidate_pipeline.py'),
             'finalize','--run-dir',str(ROOT),'--packet',str(packet),'--dispatch',str(path)]
    subprocess.run(command,check=True)
    print(json.dumps({'packet':packet.name,'output_sha256':sha,'counts':json.loads((packet/'final_manifest.json').read_text())['counts']}))

import json,urllib.request
from pathlib import Path
R=Path('/workspace/h3-continuity-research');out={}
for p in sorted(R.glob('*.job.json')):
 if not (p.name.startswith('memory_') or p.name.startswith('long_context_')):continue
 job=json.loads(p.read_text());pid=job['prompt_id']
 h=json.load(urllib.request.urlopen('http://127.0.0.1:18818/history/'+pid)).get(pid)
 if not h:continue
 (R/(p.name.replace('.job.','.history.'))).write_text(json.dumps(h,indent=2))
 messages=h['status']['messages'];ts={k:v.get('timestamp') for k,v in messages}
 out[p.stem.removesuffix('.job')]={'status':h['status']['status_str'],'seconds':(ts['execution_success']-ts['execution_start'])/1000 if 'execution_success' in ts else None,'outputs':h['outputs']}
(R/'memory_run_results.json').write_text(json.dumps(out,indent=2))
for k,v in out.items():print(k,v['status'],v['seconds'])

"""Queue one saved API graph in an already running ComfyUI; does not install models."""
import argparse,json,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('workflow',type=Path);p.add_argument('--url',default='http://127.0.0.1:8188');args=p.parse_args()
graph=json.loads(args.workflow.read_text())
req=urllib.request.Request(args.url.rstrip('/')+'/prompt',data=json.dumps({'prompt':graph}).encode(),headers={'Content-Type':'application/json'})
try:
 with urllib.request.urlopen(req) as r:print(r.read().decode())
except urllib.error.HTTPError as e:
 raise SystemExit(e.read().decode())

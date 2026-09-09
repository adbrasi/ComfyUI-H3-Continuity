"""Restore exact demo inputs without overwriting different files."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy-root',type=Path,required=True);args=p.parse_args();root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for name,digest in manifest['sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Checksum mismatch: '+name)
subprocess.run([sys.executable,str(root.parent/'research_memory_01/restore_inputs.py'),'--comfy-root',str(args.comfy_root)],check=True)
for name,destination in manifest['inputs'].items():
 source=root/'artifacts'/name;target=args.comfy_root/destination
 if target.exists():
  if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():raise SystemExit('Different file exists: '+str(target))
  print('Already present:',target);continue
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);print('Restored:',target)

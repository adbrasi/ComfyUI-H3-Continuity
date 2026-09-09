"""Restore the exact source latents for the memory experiment (standard library only)."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy-root',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for filename,digest in manifest['sha256'].items():
 file=root/filename
 if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise SystemExit('Checksum mismatch: '+filename)
for name,destination in manifest['inputs'].items():
 source=root/'artifacts'/name;target=args.comfy_root/destination
 if target.exists():
  if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():raise SystemExit('Different file already exists: '+str(target))
  print('Already present:',target);continue
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);print('Restored:',target)

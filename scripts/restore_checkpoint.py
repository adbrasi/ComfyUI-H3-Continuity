"""Restore Marco 01 inputs without replacing existing files. Standard library only."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def restore(comfy_root):
    pack = Path(__file__).resolve().parents[1]
    milestone = pack / 'milestones' / 'marco_01'
    manifest = json.loads((milestone / 'manifest.json').read_text())
    artifacts = milestone / 'artifacts'
    destinations = {
        '01_source.safetensors': comfy_root / 'output/h3_continuity_checkpoints/marco_01/01_source.safetensors',
        '02_bolsa.safetensors': comfy_root / 'output/h3_continuity_checkpoints/marco_01/02_bolsa.safetensors',
        '03_porta.safetensors': comfy_root / 'output/h3_continuity_checkpoints/marco_01/03_porta.safetensors',
        '02_bolsa.mp4': comfy_root / 'input/h3_m01_bolsa.mp4',
        '03_porta.mp4': comfy_root / 'input/h3_m01_porta.mp4',
    }
    for name, dest in destinations.items():
        source = artifacts / name
        expected = manifest['artifacts'][name]['sha256']
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {source}')
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
                raise FileExistsError(f'Preserving different existing file: {dest}')
            print(f'Already restored: {dest}')
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        print(f'Restored: {dest}')
    missing = [name for name in manifest['models'] if not (comfy_root / 'models' / name).exists()]
    if missing:
        print('Select equivalent installed model names in the workflow, or provide these files:')
        for name in missing:
            print('  ' + name)
    print('Open milestones/marco_01/03_porta.json in ComfyUI to reproduce the last continuation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True, help='Path to your ComfyUI directory')
    args = parser.parse_args()
    restore(args.comfy_root.resolve())

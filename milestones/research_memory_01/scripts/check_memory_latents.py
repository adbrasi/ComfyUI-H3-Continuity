import json
from pathlib import Path
from safetensors.torch import load_file
D=Path('/workspace/H3-testes');results={}
source=load_file(str(D/'memory_source.safetensors'))['video']
for p in sorted(D.glob('memory_*.safetensors')):
 if 'source' in p.name:continue
 z=load_file(str(p))['video']
 steps=27 if any(w in p.name for w in ['tail90','guide90','anchors90','pinned_pixels90']) else 7
 results[p.stem]={'target_latent_steps':z.shape[2],'prefix_max_error_vs_raw_source':float((z[:,:,:steps].float()-source[:,:,-steps:].float()).abs().max())}
z=load_file(str(D/'long_context_112_requested.safetensors'))['video'];sv=load_file(str(D/'source.safetensors'))['video']
results['long_context_112_requested']={'used_frames':107,'new_frames':102,'target_frames':209,'joined_frames':226,'prefix_max_error_vs_raw_source':float((z[:,:,:32].float()-sv[:,:,-32:].float()).abs().max())}
(D/'memory_latent_metrics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

import pathlib,json,sys,importlib.util,requests
from safetensors.torch import load_file,save_file
REPO=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,'/workspace/comfy/ComfyUI');import comfy.cli_args;comfy.cli_args.args.cpu=True
s=importlib.util.spec_from_file_location('h3_continuity',REPO/'__init__.py',submodule_search_locations=[str(REPO)]);pkg=importlib.util.module_from_spec(s);sys.modules[s.name]=pkg;s.loader.exec_module(pkg)
from h3_continuity.latent_assembly import H3ContinuityLatentAssemble
from comfy.nested_tensor import NestedTensor
OUT=pathlib.Path('/workspace/comfy/ComfyUI/output/h3_action_validation');x=load_file(OUT/'jungle/source.safetensors');v,a=x['video'],x['audio']
for i in range(1,4):
 x=load_file(OUT/'jungle'/f'chain_jungle_hard_{i:02d}.safetensors');z=H3ContinuityLatentAssemble().assemble({'samples':NestedTensor([v,a])},{'samples':NestedTensor([x['video'],x['audio']])},{'context_frames':39,'target_frames':141,'handover_frame':39})[0];v,a=z['samples'].unbind()
save_file({'video':v,'audio':a},str(OUT/'jungle/canonical430.safetensors'),metadata={'h3_continuity_version':'1','fps':'24','end_seconds':str(430/24)})
def submit(name,g):
 (HERE/'api'/(name+'.api.json')).write_text(json.dumps(g,indent=2)+'\n');r=requests.post('http://127.0.0.1:8188/prompt',json={'prompt':g}).json();print(name,r,flush=True);(HERE/'api'/(name+'.submit.json')).write_text(json.dumps(r,indent=2))
base=json.loads((HERE/'api/jungle_source.api.json').read_text());g=json.loads(json.dumps(base));g['8']['inputs']['noise_seed']=26601
g['20']={'class_type':'H3ContinuityLoad','inputs':{'filename':'h3_action_validation/jungle/source.safetensors'}};g['22']={'class_type':'VAEDecode','inputs':{'samples':['20',0],'vae':['3',0]}};g['40']={'class_type':'ImageFromBatch','inputs':{'image':['22',0],'batch_index':0,'length':1}};g['5']['inputs']['first_frame']=['40',0]
for node,key in [('15','filename_prefix'),('16','filename'),('17','filename_prefix')]:g[node]['inputs'][key]='h3_action_validation/jungle/fresh'+('.safetensors' if node=='16' else '')
submit('jungle_fresh',g)
g=json.loads((REPO/'milestones/gpu_validation/executed/bridge_after8_repair.api.json').read_text());g['20']['inputs']['filename']='h3_action_validation/jungle/canonical430.safetensors';g['42']['inputs']['filename']='h3_action_validation/jungle/fresh.safetensors';g['5']['inputs']['prompt']=base['5']['inputs']['prompt'];g['8']['inputs']['noise_seed']=27601;g['40']={'class_type':'ImageFromBatch','inputs':{'image':['22',0],'batch_index':0,'length':1}}
for node,key in [('26','filename_prefix'),('16','filename'),('30','filename_prefix'),('41','filename')]:g[node]['inputs'][key]='h3_action_validation/jungle/bridge'+('.cond.safetensors' if node=='41' else '.safetensors' if node=='16' else '')
g['41']['inputs']['settings']=json.dumps({'seed':27601,'steps':8,'target_frames':192,'context_frames':39,'feather_frames':17,'feather_strength':1.0})
submit('jungle_bridge',g)
g=json.loads((REPO/'workflows/06_retake.api.json').read_text());g['20']['inputs']['filename']='h3_action_validation/combat/chain_combat_hard_02.safetensors';g['5']['inputs']['length']=141;g['5']['inputs']['prompt']=json.loads((HERE/'scenarios.json').read_text())['combat']['prompt']+' During the middle interval the navy fighter steps aside and performs one clear parry of the burgundy fighters sword strike, then both resume the same sparring pace. Preserve both distinct outfits and all courtyard architecture.'
g['21']['inputs'].update(start_frame=22,end_frame=73,grow_frames=4,feather_frames=8);g['8']['inputs']['noise_seed']=38702
g['40']={'class_type':'ImageFromBatch','inputs':{'image':['22',0],'batch_index':0,'length':1}};g['41']={'class_type':'ImageFromBatch','inputs':{'image':['22',0],'batch_index':140,'length':1}};g['5']['inputs']['first_frame']=['40',0];g['5']['inputs']['last_frame']=['41',0]
for node,key in [('15','filename_prefix'),('16','filename'),('17','filename_prefix'),('29','filename_prefix')]:g[node]['inputs'][key]='h3_action_validation/combat/retake'+('.safetensors' if node=='16' else '_mask' if node=='29' else '')
submit('combat_retake',g)

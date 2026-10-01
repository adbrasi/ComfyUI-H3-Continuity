import json,pathlib,requests
REPO=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parent
baseline=json.loads((REPO/'milestones/action_validation/api/combat_feather.api.json').read_text());prompt=baseline['5']['inputs']['prompt']
def submit(name,g):
 (HERE/'api'/(name+'.api.json')).write_text(json.dumps(g,indent=2)+'\n');x=requests.post('http://127.0.0.1:8188/prompt',json={'prompt':g}).json();print(name,x,flush=True);(HERE/'api'/(name+'.submit.json')).write_text(json.dumps(x,indent=2)+'\n')
for method in ['pinned_prefix','anchors']:
 g=json.loads(json.dumps(baseline))
 for n in g.values():
  if n['class_type']=='H3ContinuityPrepare':n['inputs']['method']=method;n['inputs']['feather_frames']=17 if method!='anchors' else 0;n['inputs']['feather_strength']=1.
  for key in ['filename','filename_prefix']:
   if key in n['inputs'] and n['class_type']!='H3ContinuityLoad':n['inputs'][key]=n['inputs'][key].replace('h3_action_validation/combat/','h3_method_validation/').replace('combat_feather',method)
 submit(method,g)
g=json.loads((REPO/'milestones/action_validation/api/combat_source.api.json').read_text());g['8']['inputs']['noise_seed']=35704
g['20']={'class_type':'H3ContinuityLoad','inputs':{'filename':'h3_action_validation/combat/source.safetensors'}};g['22']={'class_type':'VAEDecode','inputs':{'samples':['20',0],'vae':['3',0]}};g['40']={'class_type':'ImageFromBatch','inputs':{'image':['22',0],'batch_index':0,'length':1}};g['5']['inputs']['first_frame']=['40',0]
for node,key in [('15','filename_prefix'),('16','filename'),('17','filename_prefix')]:g[node]['inputs'][key]='h3_method_validation/future'+('.safetensors' if node=='16' else '')
submit('bridge_future',g)
g=json.loads((REPO/'milestones/action_validation/api/jungle_bridge.api.json').read_text());g['5']['inputs'].update(prompt=prompt,length=260);g['8']['inputs']['noise_seed']=35702;g['20']['inputs']['filename']='h3_action_validation/combat/source.safetensors';g['42']['inputs']['filename']='h3_method_validation/future.safetensors'
for node,key in [('26','filename_prefix'),('16','filename'),('30','filename_prefix'),('41','filename')]:g[node]['inputs'][key]='h3_method_validation/bridge'+('.cond.safetensors' if node=='41' else '.safetensors' if node=='16' else '')
g['41']['inputs']['settings']=json.dumps({'seed':35702,'steps':8,'target_frames':260,'context_frames':39,'feather_frames':17,'feather_strength':1.0,'future_seed':35704})
submit('bridge',g)
(HERE/'api/pinned_av_control.api.json').write_text(json.dumps(baseline,indent=2)+'\n')
(HERE/'design.json').write_text(json.dumps({'source':'h3_action_validation/combat/source.safetensors','reference':'source first decoded frame','resolution':[736,416],'fps':24,'source_frames':124,'output_frames':430,'context_frames':39,'audio_context_seconds':1,'feather_frames':17,'feather_strength':1,'chain_seeds':[35702,35703,35704],'control':'existing pinned_av feather video from action validation','anchors_feather':'not supported; zero and explicitly labeled','bridge':'source124 + generated middle182 + independent future124 =430; target260 with two context windows39; not equal sampling budget or trajectory','retake':'not part of this experiment'},indent=2)+'\n')

"""Three retake approaches, same latent/prompt/seed; feather on throughout."""
import pathlib,json,requests,copy
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parents[1]
b=json.loads((REPO/'milestones/action_validation/api/combat_retake.api.json').read_text())
b['20']['inputs']['filename']='h3_action_validation/combat/chain_combat_feather_02.safetensors'
b['21']['inputs'].update(start_frame=34,end_frame=107,grow_frames=0,feather_frames=8,video_strength=1.,edit_audio=False)
b['8']['inputs']['noise_seed']=58702
b['5']['inputs']['prompt']='One uninterrupted photorealistic fantasy action shot of the same two adult swordsmen sparring in the same sunlit ancient stone courtyard, same navy blue tunic with brown leather armor on the dark haired fighter, same burgundy tunic and dark leather armor on the light haired fighter. Begin in exactly the original normal sword duel. During the middle interval both steel sword blades magically transform into brilliant glowing energy swords, cyan for the navy fighter and hot magenta for the burgundy fighter. They exchange spectacular glowing sword strikes and blocks with vivid blue and pink sparks. A large luminous swirling orange circular magic portal briefly opens directly between and behind them in the courtyard, pulsing with golden light. This is an unmistakable magical event within the existing duel. Toward the end the portal closes and the energy blades transform back into the original steel swords, and both fighters resume the original sparring pose and pace. Keep both faces, tunics, anatomy and courtyard architecture consistent. Same continuous smooth camera arc, same camera distance, no cuts or teleportation. Preserve original audio.'
# Also deliver an exact decode of this latent for direct frame comparisons.
b['50']={'class_type':'CreateVideo','inputs':{'images':['22',0],'audio':['23',0],'fps':24}}
b['51']={'class_type':'SaveVideo','inputs':{'video':['50',0],'filename_prefix':'h3_retake_validation/source','format':'auto','codec':'auto'}}
for name in ['temporal','differential','spatial']:
 g=copy.deepcopy(b)
 if name!='temporal':
  g['60']={'class_type':'H3RetakeDifferentialDiffusion','inputs':{'model':['6',0],'latent':['21',0],'sigmas':['10',0],'strength':1.}}
  g['7']['inputs']['model']=['60',0]
 if name=='spatial':
  g['61']={'class_type':'SolidMask','inputs':{'value':0.,'width':736,'height':416}}
  g['62']={'class_type':'SolidMask','inputs':{'value':1.,'width':544,'height':304}}
  g['63']={'class_type':'MaskComposite','inputs':{'destination':['61',0],'source':['62',0],'x':96,'y':112,'operation':'add'}}
  g['21']['inputs']['mask']=['63',0]
 for node,key in [('15','filename_prefix'),('16','filename'),('17','filename_prefix'),('29','filename_prefix')]:
  g[node]['inputs'][key]='h3_retake_validation/'+name+('.safetensors' if node=='16' else '_mask' if node=='29' else '')
 (HERE/'api'/f'{name}.api.json').write_text(json.dumps(g,indent=2)+'\n')
 r=requests.post('http://127.0.0.1:8188/prompt',json={'prompt':g});r.raise_for_status();response=r.json()
 (HERE/'api'/f'{name}.submit.json').write_text(json.dumps(response,indent=2)+'\n');print(name,response,flush=True)
(HERE/'design.json').write_text(json.dumps({'source':b['20']['inputs']['filename'],'frames':141,'fps':24,'interval':[34,107],'feather':8,'seed':58702,'strength':1,'edit_audio':False,'prompt':b['5']['inputs']['prompt'],'spatial_mask':{'x':96,'y':112,'width':544,'height':304},'methods':['temporal soft mask','temporal differential diffusion','spatial differential diffusion']},indent=2)+'\n')

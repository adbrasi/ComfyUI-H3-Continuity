"""Render one labeled panel per extension method; no feather-off duplicates."""
import json, pathlib, subprocess
HERE=pathlib.Path(__file__).resolve().parent
OUT=pathlib.Path('/workspace/comfy/ComfyUI/output')
cases=[
 ('01 PINNED AV | feather ON',OUT/'h3_action_validation/combat/chain_combat_feather_feather_00001_.mp4'),
 ('02 PINNED PREFIX | feather ON',OUT/'h3_method_validation/chain_pinned_prefix_feather_00001_.mp4'),
 ('03 ANCHORS | feather N/A',OUT/'h3_method_validation/chain_anchors_feather_00001_.mp4'),
 ('04 BRIDGE + NOVA ANCORA | feather ON',OUT/'h3_method_validation/bridge_00002_.mp4'),
]
cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning']
filters=[]
for i,(label,path) in enumerate(cases):
 cmd+=['-i',str(path)]
 filters.append(f"[{i}:v]fps=24,trim=end_frame=430,setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=960:580:(ow-iw)/2:40+(540-ih)/2:black,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{label}':x=16:y=10:fontsize=23:fontcolor=white[v{i}]")
filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_580|960_580[v]')
dest=OUT/'h3_method_validation/comparacao_metodos_2x2.mp4'
cmd+=['-filter_complex',';'.join(filters),'-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
subprocess.run(cmd,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)]))
(HERE/'grid.json').write_text(json.dumps({'output':str(dest),'panels':[{'label':l,'input':str(p)} for l,p in cases],'probe':probe},indent=2)+'\n')
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-ss','10','-i',str(dest),'-frames:v','1',str(HERE/'previews/grid_10sec.jpg')],check=True)
print(dest)

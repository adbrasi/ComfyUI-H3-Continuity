"""Compare retakes and insert their edited interval into the approved full duel."""
import pathlib,subprocess,json
HERE=pathlib.Path(__file__).resolve().parent;OUT=pathlib.Path('/workspace/comfy/ComfyUI/output/h3_retake_validation')
ORIGINAL=pathlib.Path('/workspace/comfy/ComfyUI/output/h3_action_validation/combat/chain_combat_feather_feather_00001_.mp4')
# Exact native support for requested [34,107), feather8: [34,107).
# Second extension's local frame0 maps to full-timeline frame187.
START,END,OFFSET=34,107,187
names=['temporal','differential','spatial']
paths={name:OUT/f'{name}_00002_.mp4' for name in names}
for name,path in paths.items():
 dest=OUT/f'{name}_duelo_completo.mp4'
 filt=f'[0:v]trim=end_frame={OFFSET+START},setpts=PTS-STARTPTS[a];[1:v]trim=start_frame={START}:end_frame={END},setpts=PTS-STARTPTS[b];[0:v]trim=start_frame={OFFSET+END}:end_frame=430,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0[v]'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(ORIGINAL),'-i',str(path),'-filter_complex',filt,'-map','[v]','-map','0:a:0','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(dest)],check=True)

def grid(full):
 labels=['01 ORIGINAL','02 RETAKE TEMPORAL | feather ON','03 TEMPORAL + DD | feather ON','04 ESPACIAL + DD | feather ON']
 inputs=[ORIGINAL if full else OUT/'source_00001_.mp4']+[OUT/f'{n}_duelo_completo.mp4' if full else paths[n] for n in names]
 total=430 if full else 141;lo=(OFFSET+START if full else START)/24;hi=(OFFSET+END if full else END)/24
 cmd=['ffmpeg','-y','-hide_banner','-loglevel','error'];filters=[]
 for i,(label,path) in enumerate(zip(labels,inputs)):
  cmd+=['-i',str(path)]
  filters.append(f"[{i}:v]fps=24,trim=end_frame={total},setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=960:610:(ow-iw)/2:40+(540-ih)/2:black,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{label}':x=16:y=10:fontsize=23:fontcolor=white,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='INTERVALO DO RETAKE':x=16:y=583:fontsize=19:fontcolor=yellow:enable='gte(t,{lo})*lt(t,{hi})'[v{i}]")
 filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_610|960_610[v]')
 dest=OUT/('comparacao_retake_duelo_completo_2x2.mp4' if full else 'comparacao_retake_trecho_2x2.mp4')
 cmd+=['-filter_complex',';'.join(filters),'-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
 subprocess.run(cmd,check=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)]));assert int(probe['streams'][0]['nb_frames'])==total
 (HERE/('grid_full.json' if full else 'grid_short.json')).write_text(json.dumps({'output':str(dest),'panels':list(zip(labels,map(str,inputs))),'effective_local_support':[START,END],'offset':OFFSET,'probe':probe},indent=2)+'\n')
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-ss',str(10.5 if full else 3),'-i',str(dest),'-frames:v','1',str(HERE/'previews'/('grid_full.jpg' if full else 'grid_short.jpg'))],check=True)
 print(dest,flush=True)
grid(False);grid(True)

#!/usr/bin/env python3
"""Render the twelve action experiments into one labeled, silent 4x3 mosaic."""
import argparse,json,math,subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/workspace/comfy/ComfyUI/output/h3_action_validation'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve();output=args.output or root/'todos_testes_4x3.mp4'
    specs=[
      ('Selva | fonte','jungle/source_*.mp4'),
      ('Selva | 3 extensoes','jungle/chain_jungle_hard_hard_*.mp4'),
      ('Selva | ancora nova','jungle/fresh_*.mp4'),
      ('Selva | bridge','jungle/bridge_*.mp4'),
      ('Duelo | fonte','combat/source_*.mp4'),
      ('Duelo | mascara fixa','combat/chain_combat_hard_hard_*.mp4'),
      ('Duelo | feather 17, forca 1','combat/chain_combat_feather_feather_*.mp4'),
      ('Duelo | retake isolado','combat/retake_*.mp4'),
      ('Neve | fonte','snow/source_*.mp4'),
      ('Neve | 3 extensoes','snow/chain_snow_hard_hard_*.mp4'),
      ('Duelo 1024x576 | fonte','combat_576p/source_*.mp4'),
      ('Duelo 1024x576 | 2 extensoes','combat_576p/chain_combat_576p_hard_*.mp4'),
    ]
    clips=[]
    for label,pattern in specs:
        files=list(root.glob(pattern))
        if len(files)!=1:raise ValueError(f'Expected exactly one file for {pattern}, got {len(files)}')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(files[0])]))
        clips.append((label,files[0],float(probe['format']['duration'])))
    frames=math.ceil(max(c[2] for c in clips)*24);duration=frames/24
    command=['ffmpeg','-y','-hide_banner','-loglevel','warning','-filter_complex_threads','4']
    for _,path,_ in clips:command+=['-i',str(path)]
    filters=[]
    for i,(label,_,length) in enumerate(clips):
        text=f'{i+1:02d}  {label} | {length:.1f}s'
        filters.append(f'[{i}:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=640:384:(ow-iw)/2:24+(360-ih)/2:black,setsar=1,drawtext=text=\'{text}\':fontcolor=white:fontsize=18:x=10:y=3,tpad=stop_mode=clone:stop_duration={duration},trim=end_frame={frames},setpts=PTS-STARTPTS,drawtext=text=\'FIM\':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.7:x=w-tw-10:y=h-th-10:enable=\'gte(t,{length})\'[v{i}]')
    layout='|'.join(f'{(i%4)*640}_{(i//4)*384}' for i in range(12))
    filters.append(''.join(f'[v{i}]' for i in range(12))+f'xstack=inputs=12:layout={layout}:fill=black[out]')
    command+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(command,check=True)
    manifest={'video':str(output),'layout':'4x3','resolution':[2560,1152],'fps':24,'duration':duration,'audio':False,'short_clips':'hold last frame','panels':[{'number':i+1,'label':label,'source':str(path),'duration':length} for i,(label,path,length) in enumerate(clips)]}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(output)


if __name__=='__main__':main()

from pathlib import Path
import av,json,numpy as np
from PIL import Image,ImageDraw
D=Path('/workspace/H3-testes')
results={}
for p in sorted(D.glob('memory_*_joined_*.mp4')):
 with av.open(str(p)) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
 n=len(frames);indices=[0,8,48,89,90,94,106,122,138,154,174,n-1]
 out=Image.new('RGB',(1472,684),'#171717');draw=ImageDraw.Draw(out)
 for j,i in enumerate(indices):
  x=j%4*368;y=j//4*228;out.paste(Image.fromarray(frames[i]).resize((368,208)),(x,y));draw.text((x+5,y+210),f'frame {i}'+(' | NEW' if i==90 else ''),fill='white')
 name=p.name.split('_joined_')[0];out.save(D/f'{name}_contact.jpg')
 f=np.stack(frames).astype(np.float32)/255;diff=np.abs(np.diff(f,axis=0)).mean((1,2,3))
 results[name]={'frames':n,'new_frames':n-90,'seam_ratio':float(diff[89]/max(1e-7,np.median(diff[77:102])))}
(D/'memory_visual_metrics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

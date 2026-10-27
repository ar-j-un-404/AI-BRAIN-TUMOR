import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_bag(path, expected):
    if sha(path)!=expected:raise ValueError('Cached input changed: '+str(path))
    a=np.load(path,allow_pickle=False)
    if a.shape!=(24,1,96,96) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Invalid cached input: '+str(path))
    return a

def statistics(a,b):
    return dict(original_mean=float(a.mean()),extracted_mean=float(b.mean()),
                original_std=float(a.std()),extracted_std=float(b.std()),
                original_nonzero_fraction=float(np.mean(a!=0)),extracted_nonzero_fraction=float(np.mean(b!=0)),
                mean_absolute_change=float(np.abs(a-b).mean()),
                original_nearly_empty_slices=int(sum(np.count_nonzero(s)<96 for s in a)),
                extracted_nearly_empty_slices=int(sum(np.count_nonzero(s)<96 for s in b)))

def montage(a,b,path,title):
    canvas=Image.new('RGB',(960,1120),(18,24,34));draw=ImageDraw.Draw(canvas)
    draw.text((12,8),title,fill='white')
    draw.text((12,28),'All 24 actual model slices | each pair: original ABOVE / extracted BELOW | common -5 to +5 scale',fill='white')
    indices=np.rint(np.linspace(14,81,24)).astype(int)
    for i in range(24):
        x=(i%6)*160;y=60+(i//6)*265
        draw.text((x+5,y),f'#{i+1} z={indices[i]}',fill='white')
        for j,bag in enumerate([a,b]):
            v=(np.clip((bag[i,0]+5)/10,0,1)*255).astype('uint8')
            tile=Image.fromarray(v).resize((116,116),Image.Resampling.NEAREST)
            canvas.paste(tile,(x+20,y+18+j*120))
    canvas.save(path)

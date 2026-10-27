import numpy as np

def region_mask(kind):
    mask=np.ones((96,96),dtype=bool)
    if kind=='border':mask[12:84,12:84]=False
    elif kind!='full':raise ValueError(kind)
    return mask

def features(bag,kind):
    a=np.asarray(bag)
    if a.shape!=(24,1,96,96) or not np.isfinite(a).all():
        raise ValueError('Expected finite 24 x 1 x 96 x 96 cache')
    mask=region_mask(kind)

    regions=[mask & (np.indices(mask.shape)[0]<48),
             mask & (np.indices(mask.shape)[0]>=48),
             mask & (np.indices(mask.shape)[1]<48),
             mask & (np.indices(mask.shape)[1]>=48)]
    result=[]
    for chunk in np.split(a[:,0],6):
        for region in regions:
            v=chunk[:,region].ravel()
            result.extend([v.mean(),v.std(),np.abs(v).mean(),
                           np.mean(np.abs(v)<1e-6),*np.quantile(v,[.1,.5,.9])])
    return np.asarray(result,dtype=np.float64)

def preview(bag,path,title):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(6*192,240),(20,25,35))
    draw=ImageDraw.Draw(canvas);draw.text((8,5),title,fill='white')
    for j,i in enumerate([0,4,9,14,19,23]):
        a=bag[i,0];rgb=np.repeat(np.uint8(np.clip((a+5)/10,0,1)*255)[...,None],3,axis=2)
        rgb[~region_mask('border')]=[24,45,65]
        tile=Image.fromarray(rgb).resize((192,192))
        canvas.paste(tile,(j*192,25))
    draw.text((8,222),'Blue centre excluded. Border may contain anatomy; this is NOT a verified background mask.',fill='white')
    canvas.save(path)

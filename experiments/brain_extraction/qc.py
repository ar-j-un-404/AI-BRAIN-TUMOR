import hashlib
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion,label
from PIL import Image,ImageDraw

def select_cases(records):
    selected=[r for r in records if r['source']=='BTC']
    for site in sorted({r['site'] for r in records if r['source']=='IXI'}):
        pool=[r for r in records if r['source']=='IXI' and r['site']==site]
        selected+=sorted(pool,key=lambda r:hashlib.sha256(('brain-qc-v1|'+r['id']).encode()).hexdigest())[:3]
    return selected

def prepare_input(record,destination):
    image=nib.load(record['path'])
    if len(image.shape)!=3 or image.header.get_xyzt_units()[0]!='mm':raise ValueError('Expected 3D MRI in mm')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3]))<1e-10:raise ValueError('Invalid affine')
    canonical=nib.as_closest_canonical(image)
    a=canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(a).all() or np.ptp(a)==0:raise ValueError('Invalid MRI intensities')
    plain=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    audit=hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a,dtype='<f4').tobytes()).hexdigest()
    if (audit if record['source']=='IXI' else plain)!=record['expected_hash']:
        raise ValueError('Input changed since reviewed manifest: '+record['id'])

    ras=nib.Nifti1Image(a,canonical.affine,canonical.header)
    transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(ras.affine),nib.orientations.axcodes2ornt(('L','A','S')))
    las=ras.as_reoriented(transform)
    if nib.aff2axcodes(las.affine)!=('L','A','S'):raise ValueError('LAS reorientation failed')
    nib.save(las,destination)
    return ras

def check_mask(image,mask_image):
    mask_image=nib.as_closest_canonical(mask_image)
    if mask_image.shape!=image.shape or not np.allclose(mask_image.affine,image.affine,atol=1e-4,rtol=0):
        raise ValueError('Mask/image geometry mismatch; refusing implicit resampling')
    raw=mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(raw).all() or not set(np.unique(raw)).issubset({0.,1.}):raise ValueError('Mask is not binary')
    mask=raw.astype(bool)
    if not mask.any() or mask.all():raise ValueError('Empty/full-volume mask requires review')
    cc,n=label(mask);counts=np.bincount(cc.ravel())[1:]
    volume=float(mask.sum()*abs(np.linalg.det(image.affine[:3,:3]))/1000)
    edge=bool(any(np.take(mask,j,axis=axis).any() for axis in range(3) for j in [0,-1]))
    flags=[]
    if not 350<=volume<=2000:flags.append('brain_volume_outside_broad_review_range')
    if edge:flags.append('mask_touches_acquisition_edge')
    if counts.max()/counts.sum()<.99:flags.append('substantial_disconnected_components')
    return mask,dict(brain_volume_ml=volume,mask_voxels=int(mask.sum()),mask_fraction=float(mask.mean()),
        components=int(n),largest_component_fraction=float(counts.max()/counts.sum()),
        touches_edge=edge,automatic_flags=flags,visual_review='pending',
        note='These checks do not establish tumor preservation or anatomical accuracy.')

def montage(image,mask,path,title):
    a=image.get_fdata(dtype=np.float32);v=a[a!=0]
    lo,hi=np.percentile(v,[1,99]);gray=np.uint8(np.clip((a-lo)/max(hi-lo,1e-6),0,1)*255)
    boundary=mask & ~binary_erosion(mask)
    canvas=Image.new('RGB',(1200,780),(18,24,34));draw=ImageDraw.Draw(canvas)
    draw.text((10,8),title+' | green=mask boundary; original anatomy retained in preview',fill='white')

    for row,axis in enumerate([2,1,0]):
        for col,fraction in enumerate([.25,.40,.60,.75]):
            k=int(round((a.shape[axis]-1)*fraction));g=np.rot90(np.take(gray,k,axis=axis));m=np.rot90(np.take(boundary,k,axis=axis))
            rgb=np.repeat(g[...,None],3,axis=2);rgb[m]=[50,255,90]

            axes=[j for j in range(3) if j!=axis];spacing=nib.affines.voxel_sizes(image.affine)
            tile=Image.fromarray(rgb)
            width=g.shape[1]*spacing[axes[0]];height=g.shape[0]*spacing[axes[1]]
            scale=min(290/width,215/height)
            tile=tile.resize((max(1,round(width*scale)),max(1,round(height*scale))))
            x=col*300+(300-tile.width)//2;y=35+row*245
            canvas.paste(tile,(x,y));draw.text((col*300+10,y+220),f'RAS axis {axis}, slice {k}',fill='white')
    canvas.save(path)

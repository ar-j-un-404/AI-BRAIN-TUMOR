import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output,resample_from_to
from msd_data import check_header
from common import dump,sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Explicit common-grid resampling; does not perform registration or skull stripping.')
    for name in ['flair','t1','t1ce','t2','mask']:p.add_argument('--'+name,required=True)
    p.add_argument('--output',required=True);p.add_argument('--already-aligned-skullstripped',action='store_true')
    args=p.parse_args()
    if not args.already_aligned_skullstripped:raise ValueError('Verify registration/skull stripping before resampling; this script cannot establish them')
    out=Path(args.output)
    if out.exists():raise ValueError('Output exists')
    names=['flair','t1','t1ce','t2','mask'];images={}
    for name in names:
        img=nib.load(getattr(args,name));check_header(img)
        if len(img.shape)!=3 or np.prod(img.shape)>30_000_000:raise ValueError('Expected bounded 3D files')
        if img.header.get_xyzt_units()[0]!='mm':raise ValueError('Millimetre units required')
        arr=img.get_fdata(dtype=np.float32)
        if not np.isfinite(arr).all():raise ValueError('Non-finite input')
        if name=='mask' and not np.allclose(arr,np.rint(arr)):raise ValueError('Mask is not categorical')
        images[name]=img
    anchor=images['flair']
    for name,img in images.items():
        if img.shape!=anchor.shape or not np.allclose(img.affine,anchor.affine,atol=1e-3):raise ValueError('Files must already share a registered grid')
    target=resample_to_output(anchor,voxel_sizes=(1,1,1),order=1)
    if np.prod(target.shape)>30_000_000:raise ValueError('Resampled volume exceeds memory guard')
    out.mkdir(parents=True)
    for name,img in images.items():
        res=resample_from_to(img,(target.shape,target.affine),order=0 if name=='mask' else 1)
        res.set_qform(res.affine,1);res.set_sform(res.affine,1);res.header.set_xyzt_units('mm')
        if name=='mask':res=nib.Nifti1Image(np.rint(res.get_fdata()).astype(np.int16),res.affine,res.header)
        nib.save(res,out/f'{name}.nii.gz')
    dump(out/'preprocessing.json',{'source_sha256':{n:sha(getattr(args,n)) for n in names},
        'original_affine':anchor.affine.tolist(),'original_shape':list(anchor.shape),'target_affine':target.affine.tolist(),
        'target_shape':list(target.shape),'interpolation':'linear MRI, nearest-neighbor labels','spacing_mm':[1,1,1],
        'registration_performed':False,'skull_stripping_performed':False,'label_remapping_performed':False})
    print('Resampled outputs saved. Record original label definitions in the external contract; do not infer them from numeric values.')

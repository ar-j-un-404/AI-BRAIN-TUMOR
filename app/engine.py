import gzip
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from imaging import load_volume,check_alignment,validate_image,MAX_VOXELS
from model import SmallUNet

MODALITIES=['FLAIR','T1w','t1gd','T2w']
REGIONS={'whole_tumor':(1,2,3),'tumor_core':(2,3),'enhancing_tumor':(3,)}
PROTOCOL='ras-axial-zcyx-f32-nonzero-zscore-clip5-v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen_paths(project):
    run=Path(project)/'runs/stage6_internal_test_001'
    plan=read_json(run/'evaluation_plan.json')
    path=run/'frozen_model.pt'
    if sha256(path)!=plan['checkpoint_sha256']:raise ValueError('Frozen checkpoint checksum mismatch.')
    if plan['protocol']!=PROTOCOL:raise ValueError('Unsupported model preprocessing protocol.')
    return path,plan


def load_frozen(project,device):
    path,plan=frozen_paths(project)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['configuration']['protocol']!=PROTOCOL:raise ValueError('Checkpoint preprocessing mismatch.')
    model=SmallUNet().to(device);model.load_state_dict(saved['model_state']);model.eval()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return model,plan


def prepare_inputs(paths,reference=None):
    
    paths=[Path(p) for p in paths]
    if len(paths) not in (1,4):raise ValueError('Supply one four-channel 4D NIfTI or four assigned 3D NIfTIs.')
    for p in paths:
        if not p.is_file():raise FileNotFoundError(str(p))
        if not p.name.lower().endswith(('.nii','.nii.gz')):raise ValueError('Only NIfTI .nii or .nii.gz is supported.')
    original=nib.load(paths[0]);original_shape=original.shape[:3];original_affine=original.affine.copy()
    if len(paths)==1:
        if len(original.shape)!=4 or original.shape[3]!=4:raise ValueError('The 4D file must contain exactly four channels: FLAIR, T1w, t1gd, T2w.')

        validate_image(original.slicer[:,:,:,0])
        if original.header.get_xyzt_units()[0]!='mm':raise ValueError('Millimetre spatial units are required.')
        image=nib.as_closest_canonical(original)
        affine=image.affine;spacing=nib.affines.voxel_sizes(affine)
        if not np.allclose(affine[:3,:3]/spacing,np.eye(3),atol=1e-3,rtol=0):raise ValueError('Oblique/sheared input requires reviewed preprocessing.')
        volumes=[]
        for c in range(4):
            a=np.asarray(image.dataobj[...,c],dtype=np.float32)
            volumes.append({'data':a,'affine':affine,'spacing':spacing,'unit':'mm'})
    else:
        volumes=[load_volume(p) for p in paths]
        for vol in volumes:
            if vol['unit']!='mm':raise ValueError('Millimetre spatial units are required.')
            check_alignment(volumes[0],vol)
        affine=volumes[0]['affine'];spacing=volumes[0]['spacing']
    shape=volumes[0]['data'].shape
    if min(shape[:2])<8:raise ValueError('In-plane dimensions must be at least 8 voxels.')
    normalized=np.empty((shape[2],4,shape[1],shape[0]),dtype=np.float32)
    statistics=[]
    for c,vol in enumerate(volumes):
        a=vol['data']
        if not np.isfinite(a).all():raise ValueError('Non-finite MRI intensity.')
        fg=a!=0
        if not fg.any():raise ValueError('Empty modality: '+MODALITIES[c])
        mean=float(a[fg].mean(dtype=np.float64));std=float(a[fg].std(dtype=np.float64))
        if std<1e-8:raise ValueError('Constant modality: '+MODALITIES[c])
        b=np.zeros_like(a);b[fg]=(a[fg]-mean)/std;np.clip(b,-5,5,out=b)
        normalized[:,c]=b.transpose(2,1,0)
        statistics.append({'modality':MODALITIES[c],'nonzero_mean':mean,'nonzero_std':std})
    ref=None
    if reference:
        r=load_volume(reference,is_mask=True);check_alignment(volumes[0],r)
        if not set(r['metadata']['labels']).issubset({0,1,2,3}):raise ValueError('Reference labels must use reviewed MSD mapping 0,1,2,3. Label 4 is not accepted.')
        ref=r['data'].astype(np.uint8)
    return {'images':normalized,'affine':np.asarray(affine),'spacing':np.asarray(spacing),
            'shape':shape,'original_affine':original_affine,'original_shape':original_shape,
            'reference':ref,'normalization':statistics,
            'input_sha256':[sha256(p) for p in paths],
            'reference_sha256':sha256(reference) if reference else None,'protocol':PROTOCOL}


@torch.inference_mode()
def infer(model,case,device,progress=None):
    x=case['images'];pred=np.empty((len(x),*x.shape[2:]),dtype=np.uint8)
    for start in range(0,len(x),2):
        batch=torch.from_numpy(np.array(x[start:start+2],dtype=np.float32,copy=True)).to(device)
        logits=model(batch)
        if not torch.isfinite(logits).all():raise ValueError('Model produced non-finite output.')
        pred[start:start+len(batch)]=logits.argmax(1).cpu().numpy().astype(np.uint8)
        if progress:progress(min(start+len(batch),len(x))/len(x))
    return pred.transpose(2,1,0)


def measurements(pred,case):
    ml=abs(float(np.linalg.det(case['affine'][:3,:3])))/1000
    rows=[]
    for name,values in REGIONS.items():
        p=np.isin(pred,values);row={'region':name,'predicted_volume_ml':float(p.sum()*ml)}
        if case['reference'] is not None:
            t=np.isin(case['reference'],values);tp=int((p&t).sum());pn=int(p.sum());tn=int(t.sum())
            row.update(reference_volume_ml=float(tn*ml),dice=2*tp/(pn+tn) if pn+tn else None,
                       false_positive_voxels=pn-tp,false_negative_voxels=tn-tp)
        rows.append(row)
    return rows


def mask_bytes(pred,case,native=True):
    
    affine=case['affine'];out=pred
    if native:
        transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),nib.orientations.io_orientation(case['original_affine']))
        out=nib.orientations.apply_orientation(pred,transform)
        if out.shape!=tuple(case['original_shape']):raise ValueError('Native export shape mismatch.')
        affine=case['original_affine']
    image=nib.Nifti1Image(np.asarray(out,dtype=np.uint8),affine)
    image.set_qform(affine,1);image.set_sform(affine,1);image.header.set_xyzt_units('mm')
    return gzip.compress(image.to_bytes(),mtime=0)

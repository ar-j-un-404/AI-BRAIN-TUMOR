import hashlib
from itertools import product
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

FOV_MM = 320.0
SIZE = 96
SPACING = FOV_MM / SIZE
PROTOCOL = 'ras-image-center-320mm-96grid-linear-zscore-native-clip5-v2'

def load_volume(path, expected_hash=None):
    image = nib.load(path)
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError(f'Expected millimetres: {path}')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-10:
        raise ValueError(f'Invalid affine: {path}')
    image = nib.as_closest_canonical(image)
    a = image.get_fdata(dtype=np.float32)
    if a.ndim != 3 or not np.isfinite(a).all() or np.ptp(a) == 0:
        raise ValueError(f'Invalid 3D MRI: {path}')
    raw = np.ascontiguousarray(a).tobytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    audit_hash = hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a,dtype='<f4').tobytes()).hexdigest()
    if expected_hash is not None and audit_hash != expected_hash:
        raise ValueError(f'Voxel content changed: {path}')
    return image, a, fingerprint

def physical_volume(image, a):
    
    shape = np.array(a.shape)
    center, extent = coverage(image)
    target = np.diag([SPACING,SPACING,SPACING,1.])
    target[:3,3] = center - SPACING*(SIZE-1)/2
    if np.any(extent > FOV_MM/2 + .001):
        raise ValueError('Acquisition exceeds fixed 320 mm field; no silent crop. Half extents: '+str(extent))
    mask = a != 0
    values = a[mask]
    if len(values)<1000 or values.std()<1e-8:
        raise ValueError('Insufficient intensity variation')
    normalized = np.zeros_like(a)
    normalized[mask] = np.clip((values-values.mean())/values.std(),-5,5)
    source = nib.Nifti1Image(normalized,image.affine)
    volume = resample_from_to(source,((96,96,96),target),order=1,mode='constant',cval=0).get_fdata(dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError('Nonfinite resampled volume')
    meta = dict(protocol=PROTOCOL, native_shape=list(map(int,shape)),
                native_spacing_mm=nib.affines.voxel_sizes(image.affine).tolist(),
                center_world_mm=center.tolist(), target_affine=target.tolist(),
                target_spacing_mm=[SPACING]*3, target_fov_mm=[FOV_MM]*3,
                acquired_half_extents_mm=extent.tolist(), coverage_check_passed=True)
    return volume, meta

def prepare_physical(path, expected_hash=None):
    image,a,fingerprint = load_volume(path,expected_hash)
    volume,meta = physical_volume(image,a)
    indices = np.rint(np.linspace(14,81,24)).astype(int)
    bag = np.ascontiguousarray(volume[:,:,indices].transpose(2,0,1)[:,None])
    return bag,fingerprint,meta


def coverage(image):
    
    if len(image.shape)!=3 or image.header.get_xyzt_units()[0]!='mm':
        raise ValueError('Expected 3D MRI with millimetre units')
    affine=image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-10:
        raise ValueError('Invalid affine')
    shape=np.array(image.shape)
    center=nib.affines.apply_affine(affine,(shape-1)/2)
    corners=np.array(list(product(*[(-.5,float(n)-.5) for n in shape])))
    world=nib.affines.apply_affine(affine,corners)
    return center,np.max(np.abs(world-center),axis=0)


def preflight(records):
    rows=[]
    for i,r in enumerate(records,1):
        row=dict(id=r['id'],path=r['path'])
        try:
            center,extent=coverage(nib.load(r['path']))
            row.update(full_extents_mm=(2*extent).tolist(),fits=bool(np.all(extent<=FOV_MM/2+.001)))
        except Exception as exc:
            row.update(fits=False,error=str(exc))
        rows.append(row)
        if i%50==0 or i==len(records):print(f'Header coverage check: {i}/{len(records)}',flush=True)
    return rows

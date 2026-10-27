import hashlib
import random
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROTOCOL = 'ras-nonzero-zscore-clip5-voxel-cube-resize64-v1'

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Classifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        layers, channels = [], 1
        for width in (8,16,32):
            layers += [nn.Conv2d(channels,width,3,padding=1),
                       nn.GroupNorm(4,width),nn.ReLU(),nn.MaxPool2d(2)]
            channels=width
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Linear(32,1)

    def forward(self,x):

        b,s,c,h,w=x.shape
        embedding=self.features(x.reshape(b*s,c,h,w)).reshape(b,s,32)
        return self.head(embedding.mean(dim=1)).squeeze(1)

def prepare_volume(path, size=64, expected_hash=None):
    image = nib.load(path)
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError(f'Expected spatial units mm: {path}')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-10:
        raise ValueError(f'Invalid affine: {path}')
    image = nib.as_closest_canonical(image)
    a = image.get_fdata(dtype=np.float32)
    if a.ndim != 3 or not np.isfinite(a).all() or np.ptp(a) == 0:
        raise ValueError(f'Invalid 3D MRI: {path}')
    fingerprint = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    audit_hash = hashlib.sha256(str(a.shape).encode() + np.ascontiguousarray(a, dtype='<f4').tobytes()).hexdigest()
    if expected_hash is not None and audit_hash != expected_hash:
        raise ValueError(f'Voxel content changed since audit: {path}')
    mask = a != 0
    values = a[mask]
    if values.size < 1000 or values.std() < 1e-8:
        raise ValueError(f'Insufficient variation: {path}')
    out = np.zeros_like(a)
    out[mask] = np.clip((values-values.mean()) / values.std(), -5, 5)
    side = max(out.shape)
    out = np.pad(out, [((side-n)//2, side-n-(side-n)//2) for n in out.shape])
    x = F.interpolate(torch.from_numpy(out)[None,None], size=(size,size,size),
                      mode='trilinear', align_corners=False)[0]
    return x, fingerprint

def prepare_bag(path,size=96,expected_hash=None):
    volume,fingerprint=prepare_volume(path,size=size,expected_hash=expected_hash)

    indices=torch.linspace(14,81,24).round().long()
    bag=volume[0,:,:,indices].permute(2,0,1).unsqueeze(1).contiguous()
    return bag,fingerprint

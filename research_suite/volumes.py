from collections import OrderedDict
from itertools import product
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from networks import enable_mc_dropout

class Patches(Dataset):
    def __init__(self, folders, patch, channels, steps, seed, epoch, augment=True):
        self.folders = list(map(Path, folders)); self.patch = tuple(patch); self.channels = channels
        self.steps = steps; self.seed = seed; self.epoch = epoch; self.augment = augment; self.cache = OrderedDict()
    def __len__(self): return self.steps
    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        folder = self.folders[int(rng.integers(len(self.folders)))]
        if folder not in self.cache:
            self.cache[folder] = (np.load(folder/'images.npy', mmap_mode='r'), np.load(folder/'masks.npy', mmap_mode='r'))
            if len(self.cache)>2: self.cache.popitem(last=False)
        self.cache.move_to_end(folder); x, y = self.cache[folder]
        center = [int(rng.integers(n)) for n in y.shape]

        if rng.random()<.5:
            valid_z = np.flatnonzero(np.any(y > 0, axis=(1, 2)))
            if len(valid_z):
                z = int(rng.choice(valid_z)); coords = np.argwhere(y[z]>0); yy, xx = coords[int(rng.integers(len(coords)))]
                center = [z, int(yy), int(xx)]
        starts = [min(max(c-p//2, 0), max(n-p, 0)) for c, p, n in zip(center, self.patch, y.shape)]
        slices = tuple(slice(s, min(s+p, n)) for s, p, n in zip(starts, self.patch, y.shape))
        a = np.array(x[slices[0], :, slices[1], slices[2]], dtype=np.float32).transpose(1,0,2,3)[self.channels]
        b = np.array(y[slices], dtype=np.int64)
        pads = [(0, p-n) for p,n in zip(self.patch, b.shape)]
        a = np.pad(a, [(0,0)]+pads); b = np.pad(b, pads)
        if self.augment:
            for axis in range(3):
                if rng.random()<.5: a = np.flip(a, axis+1); b = np.flip(b, axis)
            fg = a != 0
            scales = rng.uniform(.9, 1.1, (len(a),1,1,1))
            a = np.where(fg, np.clip(a*scales+rng.normal(0,.03,a.shape), -5,5), 0)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(b))

def starts(n, patch, overlap=.5):
    if n <= patch: return [0]
    step = max(1, int(patch*(1-overlap)))
    return sorted(set(list(range(0,n-patch+1,step))+[n-patch]))

@torch.inference_mode()
def sliding(model, images, device, patch=(64,96,96), channels=(0,1,2,3), mc=False):
    
    if mc: enable_mc_dropout(model)
    else: model.eval()
    shape = (images.shape[0], images.shape[2], images.shape[3])
    probs = np.zeros((4, *shape), np.float32); count = np.zeros(shape, np.float32)
    for start in product(*(starts(n,p) for n,p in zip(shape, patch))):
        lengths = tuple(min(p, n-s) for s,p,n in zip(start,patch,shape))
        sl = tuple(slice(s,s+l) for s,l in zip(start,lengths))
        x = np.array(images[sl[0], :, sl[1], sl[2]], dtype=np.float32).transpose(1,0,2,3)[list(channels)]
        x = np.pad(x, [(0,0)]+[(0,p-l) for p,l in zip(patch,lengths)])
        tensor = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)

        logits = model(tensor)
        if not torch.isfinite(logits).all(): raise ValueError('Non-finite prediction')
        out = logits.softmax(1)[0].cpu().numpy()[(slice(None),)+tuple(slice(0,l) for l in lengths)]
        probs[(slice(None),)+sl] += out; count[sl] += 1
    if np.any(count==0): raise RuntimeError('Sliding-window coverage hole')
    return probs/count[None]

@torch.inference_mode()
def uncertainty_maps(model, images, device, patch, channels, passes=4):

    mean = None; mean_entropy = None
    for _ in range(passes):
        p = sliding(model, images, device, patch, channels, mc=True)
        entropy = -(p*np.log(np.maximum(p,1e-8))).sum(0)/np.log(4)
        if mean is None: mean = p/passes; mean_entropy = entropy/passes
        else: mean += p/passes; mean_entropy += entropy/passes
    model.eval()
    entropy = -(mean*np.log(np.maximum(mean,1e-8))).sum(0)/np.log(4)
    return mean, entropy, np.maximum(entropy-mean_entropy, 0)

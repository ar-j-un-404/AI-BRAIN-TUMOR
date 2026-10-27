import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from common import load_checkpoint,device,dump,sha,seed_all
from classify import load_pair
from networks import Classifier3D

def gradcam(model,x):
    activations=[]
    def hook(module,args,output): activations.append(output)
    handle=model.features[-1].register_forward_hook(hook)
    try:
        model.eval(); logit=model(x)
        grads=torch.autograd.grad(logit.sum(),activations[0])[0]
        weights=grads.mean((2,3,4),keepdim=True)
        cam=(weights*activations[0]).sum(1,keepdim=True).relu()
        cam=F.interpolate(cam,size=x.shape[2:],mode='trilinear',align_corners=False)[0,0]
        cam=cam.detach().cpu().numpy(); maximum=float(cam.max())
        return cam/maximum if maximum>0 else cam
    finally: handle.remove()

if __name__=='__main__':
    p=argparse.ArgumentParser(description='3D classifier Grad-CAM and head-randomization sensitivity, in the model input grid.'); p.add_argument('--checkpoint',required=True)
    p.add_argument('--t1',required=True); p.add_argument('--t2',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); out=Path(args.output)
    if out.exists(): raise ValueError('Output exists')
    ck=load_checkpoint(args.checkpoint); model=Classifier3D(len(ck['config']['input_indices'])).to(device()); model.load_state_dict(ck['model_state'])
    x=load_pair({'t1':args.t1,'t2':args.t2})[ck['config']['input_indices']][None].to(device()); cam=gradcam(model,x)
    randomized=copy.deepcopy(model); seed_all(123); randomized.head.reset_parameters(); random_cam=gradcam(randomized,x)
    flip_cam=gradcam(model,x.flip(-1))[:,:,::-1].copy()
    def correlation(a,b):
        if np.std(a)==0 or np.std(b)==0: return None
        value=spearmanr(a.ravel(),b.ravel()).statistic; return float(value) if np.isfinite(value) else None
    out.mkdir(parents=True); np.save(out/'gradcam.npy',cam); np.save(out/'input_model_grid.npy',x[0].cpu().numpy())
    np.save(out/'randomized_head_gradcam.npy',random_cam); np.save(out/'flip_gradcam_restored.npy',flip_cam)
    dump(out/'sanity_check.json',{'checkpoint_sha256':sha(args.checkpoint),
        'head_randomization_spearman':correlation(cam,random_cam),'left_right_flip_spearman':correlation(cam,flip_cam),
        'note':'Qualitative attribution in cropped/resized 64-cubed model grid, not a tumor mask. A head-randomization check is not a complete saliency validation.'})

import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from core import prepare_bag, Classifier
from prepare import to_bag

def run():
    assert torch.cuda.is_available(), 'CUDA unavailable in main project environment'
    torch.set_num_threads(2)
    rng=np.random.default_rng(42)
    a=rng.normal(100,20,(20,24,28)).astype('float32');a[:2]=0
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'synthetic.nii.gz'
        im=nib.Nifti1Image(a,np.diag([1.,2.,3.,1.]));im.header.set_xyzt_units('mm');nib.save(im,p)
        reference,_=prepare_bag(p)
        result=to_bag(a,a!=0)
        np.testing.assert_array_equal(reference.numpy(),result)
    mask=np.zeros_like(a,dtype=bool);mask[2:18,2:22,2:26]=True
    stripped=to_bag(a,mask)
    assert stripped.shape==(24,1,96,96) and np.isfinite(stripped).all()
    model=Classifier().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    logits=model(torch.from_numpy(stripped)[None].cuda())
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones(1,device='cuda'))
    loss.backward();optimizer.step();torch.cuda.synchronize()
    print('Synthetic baseline equivalence and GPU training step PASSED. No real MRI trained.',flush=True)
if __name__=='__main__':run()

import argparse
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from contracts import read,verify_nnunet,NN_HASH,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('request');args=p.parse_args()
    req=read(args.request);out=Path(req['output']);out.mkdir(parents=True,exist_ok=True)
    os.environ['nnUNet_compile']='false'
    os.environ['OMP_NUM_THREADS']='2'
    if importlib.metadata.version('nnunetv2')!='2.8.1':raise ValueError('This adapter expects your existing nnunetv2 2.8.1 environment.')
    verify_nnunet(req['model'])
    import numpy as np
    import nibabel as nib
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the nnU-Net environment.')
    paths=[Path(x) for x in req['inputs']]
    if [sha(x) for x in paths]!=req['input_sha256']:raise ValueError('MRI changed since it was loaded. Reload it first.')
    with tempfile.TemporaryDirectory(prefix='nnunet_input_',dir=out) as td:
        files=[]
        if len(paths)==1:
            image=nib.load(paths[0])
            if image.ndim!=4 or image.shape[-1]!=4:raise ValueError('Expected four-channel MRI.')
            for c in range(4):
                f=Path(td)/f'case_{c:04d}.nii.gz';nib.save(image.slicer[:,:,:,c],f);files.append(str(f))
        elif len(paths)==4:files=[str(x) for x in paths]
        else:raise ValueError('Supply one 4D MRI or four ordered modalities.')
        print('Loading reviewed nnU-Net final checkpoint; original intensities preserved.',flush=True)
        predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,
            perform_everything_on_device=False,device=torch.device('cuda',0),verbose=False,allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(req['model'],use_folds=(0,),checkpoint_name='checkpoint_final.pth')
        predictor.predict_from_files([files],[str(out/'segmentation')],save_probabilities=False,
            overwrite=True,num_processes_preprocessing=1,num_processes_segmentation_export=1)
    result=out/'segmentation.nii.gz'
    if not result.is_file():raise FileNotFoundError('nnU-Net did not write the segmentation.')
    (out/'worker_result.json').write_text(json.dumps({'prediction':str(result),'checkpoint_sha256':NN_HASH,
        'kind':'nnU-Net 3D full-resolution','checkpoint':'checkpoint_final.pth','fold':0,'mirroring':True,
        'tile_step_size':.5,'probabilities_calibrated':False},indent=2))
    print('nnU-Net inference complete.',flush=True)

if __name__=='__main__':main()

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from contracts import sha,read,verify_fold,validate_experiment,select_folds,MODEL_REL
from engine import infer,load_frozen,PROTOCOL,SmallUNet
from common import load_checkpoint,device
from app_support import load_2d
from networks import UNet3D
from volumes import sliding,uncertainty_maps


def clear_gpu():
    if torch.cuda.is_available():torch.cuda.empty_cache()


def run_segment(kind,project,case,paths,output,progress=None,mc=False):
    project=Path(project);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if [sha(p) for p in paths]!=case['input_sha256']:raise ValueError('Input file changed after loading. Reload MRI.')
    if kind=='nnU-Net 3D (reviewed final)':
        python=project/'.venv_nnunet/Scripts/python.exe'
        if not python.is_file():raise FileNotFoundError(python)
        req={'model':str(project/MODEL_REL),'inputs':[str(p) for p in paths],
             'input_sha256':case['input_sha256'],'output':str(output)}
        request=output/'request.json';request.write_text(json.dumps(req,indent=2))
        env=os.environ.copy();env['nnUNet_compile']='false'
        workspace=project/'nnunet_reference_v1'
        for key,folder in [('nnUNet_raw','raw'),('nnUNet_preprocessed','preprocessed'),('nnUNet_results','results')]:env[key]=str(workspace/folder)
        clear_gpu()
        with (output/'inference.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([str(python),str(Path(__file__).with_name('nnunet_worker.py')),str(request)],
                                  stdout=log,stderr=subprocess.STDOUT,env=env,cwd=project)
            try:code=proc.wait()
            except BaseException:proc.terminate();proc.wait();raise
        if code:
            tail=(output/'inference.log').read_text(encoding='utf-8',errors='replace')[-4500:]
            raise RuntimeError('nnU-Net failed. Log: '+str(output/'inference.log')+'\n'+tail)
        info=read(output/'worker_result.json')
        image=nib.as_closest_canonical(nib.load(info['prediction']));pred=image.get_fdata(dtype=np.float32)
        if pred.shape!=tuple(case['shape']) or not np.allclose(image.affine,case['affine'],atol=1e-3,rtol=0):
            raise ValueError('Prediction grid mismatch. Overlay blocked; no automatic interpolation.')
        if not np.isfinite(pred).all() or not np.isin(pred,[0,1,2,3]).all():raise ValueError('Invalid predicted labels.')
        info['protocol']='official nnU-Net preprocessing from original intensities; reviewed plans'
        return pred.astype(np.uint8),info,{}
    dev=device();model=None
    try:
        if kind=='Existing 2D segmentation':
            try:model,info=load_2d(project,dev)
            except FileNotFoundError:

                if (project/'runs/stage6_internal_test_001/evaluation_plan.json').exists():raise
                path=project/'runs/stage3_full_001/best_model.pt';ck=load_checkpoint(path)
                if ck['configuration']['protocol']!=PROTOCOL:raise ValueError('Unsupported Stage 3 model protocol.')
                model=SmallUNet().to(dev);model.load_state_dict(ck['model_state']);model.eval()
                info={'kind':'Stage 3 development 2D','checkpoint':str(path),'checkpoint_sha256':sha(path)}
            pred=infer(model,case,dev,progress)
            return pred,info,{}
        if kind!='Small 3D segmentation':raise ValueError('Unknown segmentation model.')
        path=project/'runs/3d_full_all4_seed42/best.pt';ck=load_checkpoint(path);config=ck['config']
        if config['task']!='segmentation3d' or config['protocol']!='msd-normalized-native-1mm-3d-v1':raise ValueError('Unsupported 3D protocol.')
        if not np.allclose(case['spacing'],[1,1,1],atol=1e-3):raise ValueError('Small 3D model requires reviewed 1-mm isotropic scans.')
        model=UNet3D(len(config['channels']),base=config.get('base',8),dropout=config.get('dropout',.1)).to(dev)
        model.load_state_dict(ck['model_state']);model.eval()
        prob=sliding(model,case['images'],dev,config['patch'],config['channels'])
        pred=prob.argmax(0).astype(np.uint8).transpose(2,1,0)
        extra={'entropy':(-(prob*np.log(np.maximum(prob,1e-8))).sum(0)/np.log(4)).transpose(2,1,0)}
        if mc:
            _,_,mi=uncertainty_maps(model,case['images'],dev,config['patch'],config['channels'],4)
            extra['mi']=mi.transpose(2,1,0)
        return pred,{'kind':kind,'checkpoint':str(path),'checkpoint_sha256':sha(path),'epoch':ck['epoch'],'protocol':config['protocol']},extra
    finally:
        if model is not None:del model
        clear_gpu()


def classify_t1(root,variant,path):
    from classifier_core import Classifier,prepare_bag,seed_all
    exp=validate_experiment(root)
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)


    import re
    match=re.match(r'IXI(\d+)-',path.name,re.IGNORECASE)
    lock_path=Path(root).parent/'ixi_split_lock.json'
    if match:
        if not lock_path.is_file():raise FileNotFoundError('IXI split lock missing; cannot protect reserved subjects.')
        assignment=read(lock_path)['assignment'];key=str(int(match.group(1)))
        normalized={str(int(k)):v for k,v in assignment.items()}
        if normalized.get(key)!='development':raise ValueError('Reserved or unrecognized IXI participant: demo inference blocked.')
    seed_all(42);bag,fingerprint=prepare_bag(path,size=96)
    image=nib.as_closest_canonical(nib.load(path));a=image.get_fdata(dtype=np.float32)
    audit_hash=hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a,dtype='<f4').tobytes()).hexdigest()
    folds,matched=select_folds(exp['records'],fingerprint,audit_hash)

    known_names={Path(r['path'].replace('\\','/')).name:r for r in exp['records']}
    if not matched and path.name in known_names:raise ValueError('Known development filename has changed voxel content. Resolve before inference.')
    dev=device();scores=[];hashes=[]
    for fold in folds:
        checkpoint=verify_fold(root,variant,fold,exp['records'])
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ck.get('step')!=1000:raise ValueError('Checkpoint training is incomplete.')
        model=Classifier().to(dev)
        try:
            model.load_state_dict(ck['model'],strict=True);model.eval()
            with torch.inference_mode():score=float(model(bag[None].to(dev)).sigmoid().item())
            if not np.isfinite(score):raise ValueError('Nonfinite classifier score.')
            scores.append({'fold':fold,'score':score});hashes.append(sha(checkpoint))
        finally:del model;clear_gpu()
    score=float(np.mean([r['score'] for r in scores]))
    return {'task':'T1 tumor-presence research classifier','variant':variant,'predicted_class':'tumor' if score>=.5 else 'healthy',
            'score':score,'threshold':.5,'calibrated_probability':False,'fold_scores':scores,
            'matched_development_participant':matched,'evaluation_context':'Reused development held-out fold' if matched else 'Unvalidated mean of five fold models; identity independence not established',
            'input_sha256':sha(path),'checkpoint_sha256':hashes,'preprocessing':exp['preprocessing'],
            'limitations':'Exploratory BTC+IXI experiment; BTC healthy specificity was 18.18%. A score is not diagnostic confidence.'}

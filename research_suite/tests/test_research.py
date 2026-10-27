import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pytest
import torch
from common import dump,sha,rows
from networks import UNet3D,Classifier3D,segmentation_loss
from volumes import sliding,uncertainty_maps,Patches
from metrics import region_metrics,classification_metrics,risk_coverage
from classify import validate_manifest,load_pair,aggregate
from explain import gradcam
from engine import prepare_inputs,mask_bytes
from train3d import train,parser
from evaluate3d import evaluate
from msd_data import MODALITIES,LABELS

def save_nii(path,a,affine=None):
    image=nib.Nifti1Image(a,np.eye(4) if affine is None else affine)
    image.set_qform(image.affine,1);image.set_sform(image.affine,1);image.header.set_xyzt_units('mm');nib.save(image,path)

def test_real_3d_shapes_and_gradient():
    model=UNet3D();x=torch.randn(1,4,16,24,16);y=torch.randint(0,4,(1,16,24,16))
    z=model(x);assert z.shape==(1,4,16,24,16)
    loss=segmentation_loss(z,y);loss.backward();assert torch.isfinite(loss)
    assert model.e1.layers[0].weight.grad is not None

def test_sliding_coverage_and_normalization():
    class Constant(torch.nn.Module):
        def forward(self,x): return torch.zeros((len(x),4,*x.shape[2:]),device=x.device)
    x=np.ones((19,4,21,13),np.float32)
    out=sliding(Constant(),x,torch.device('cpu'),(16,16,16));assert out.shape==(4,19,21,13)
    assert np.allclose(out,.25)

def test_uncertainty_ranges():
    x=np.random.default_rng(0).normal(size=(16,4,16,16)).astype(np.float32)
    model=UNet3D(dropout=.2);p,h,mi=uncertainty_maps(model,x,torch.device('cpu'),(16,16,16),[0,1,2,3],2)
    assert np.allclose(p.sum(0),1,atol=1e-6);assert h.min()>=0 and h.max()<=1.00001
    assert mi.min()>=0 and np.all(mi<=h+1e-6);assert not model.training

def test_metric_mapping_empty_and_physical_distance():
    p=np.zeros((8,8,8),np.uint8);t=p.copy();p[2,2,2]=3;t[3,2,2]=3
    m=region_metrics(p,t,(2,1,1));assert m['whole_tumor']['hd95_mm']==2
    assert m['tumor_core']['dice']==0
    m=region_metrics(np.zeros_like(p),np.zeros_like(p));assert m['whole_tumor']['dice'] is None
    m=region_metrics(p,np.zeros_like(p));assert m['whole_tumor']['dice']==0 and m['whole_tumor']['hd95_mm'] is None
    p[:]=0;p[2,2,2]=1;m=region_metrics(p,p);assert m['whole_tumor']['dice']==1 and m['tumor_core']['dice'] is None

def test_classification_and_tied_risk():
    m=classification_metrics([0,0,1,1],[.1,.2,.7,.8]);assert m['accuracy']==1 and m['roc_auc']==1
    r=risk_coverage([0,1,0],[.5,.5,.5]);assert len(r['curve'])==1 and r['curve'][0]['n']==3

def manifest(path,records):
    keys=['case_id','participant_id','source','split','label','t1','t2','identity_evidence','protocol_evidence','label_evidence']
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in records: w.writerow({k:r.get(k,'documented') for k in keys})

def test_manifest_blocks_patient_leakage_and_source_shortcut(tmp_path):
    rs=[{'case_id':'a','participant_id':'p','source':'x','split':'train','label':'0'},
        {'case_id':'b','participant_id':'p','source':'x','split':'validation','label':'0'}]
    path=tmp_path/'m.csv';manifest(path,rs)
    with pytest.raises(ValueError,match='Participant overlaps'): validate_manifest(path,partitions=(),inspect_files=False)
    rs=[{'case_id':'a','participant_id':'p1','source':'healthy_site','split':'train','label':'0'},
        {'case_id':'b','participant_id':'p2','source':'tumor_site','split':'train','label':'1'}]
    manifest(path,rs)
    with pytest.raises(ValueError,match='Sources contain only one class'): validate_manifest(path,partitions=('train',),inspect_files=False)

def test_classifier_input_alignment_and_attribution(tmp_path):
    a=np.random.default_rng(2).normal(size=(16,16,16)).astype(np.float32);t1=tmp_path/'t1.nii.gz';t2=tmp_path/'t2.nii.gz'
    save_nii(t1,a);save_nii(t2,a*2+1);x=load_pair({'t1':t1,'t2':t2});assert x.shape==(2,64,64,64)
    model=Classifier3D();cam=gradcam(model,x[None]);assert cam.shape==(64,64,64) and np.isfinite(cam).all()
    aff=np.eye(4);aff[0,3]=10;save_nii(t2,a,aff)
    with pytest.raises(ValueError,match='same grid'):load_pair({'t1':t1,'t2':t2})

def test_preprocessing_reference_independence_and_native_export(tmp_path):
    a=np.random.default_rng(3).normal(size=(16,17,18,4)).astype(np.float32);a[:2]=0
    image=tmp_path/'i.nii.gz';mask=tmp_path/'m.nii.gz';aff=np.diag([-1,1,1,1]);aff[0,3]=15
    save_nii(image,a,aff);save_nii(mask,np.zeros(a.shape[:3],np.uint8),aff)
    c=prepare_inputs([image]);r=prepare_inputs([image],mask);assert np.array_equal(c['images'],r['images'])
    pred=np.zeros(c['shape'],np.uint8);pred[2,3,4]=3
    import gzip
    native=nib.Nifti1Image.from_bytes(gzip.decompress(mask_bytes(pred,c)))
    canonical=nib.as_closest_canonical(native);assert np.array_equal(canonical.get_fdata(),pred)
    assert np.allclose(native.affine,aff)

def test_patient_aggregation():
    r=[{'participant_id':'a','label':'1','source':'x'},{'participant_id':'a','label':'1','source':'x'},
       {'participant_id':'b','label':'0','source':'x'}]
    ids,y,z,_=aggregate(r,[1,3,-1]);assert ids==['a','b'] and z.tolist()==[2,-1]

def synthetic_project(tmp_path):
    project=tmp_path/'project';root=tmp_path/'dataset';root.mkdir();splitdir=project/'data/splits';splitdir.mkdir(parents=True)
    rng=np.random.default_rng(42);records=[]
    for i in range(3):
        cid=f'CASE_{i}';x=rng.normal(size=(16,16,16,4)).astype(np.float32);y=np.zeros((16,16,16),np.uint8)
        y[4:12,4:12,4:12]=1;y[6:10,6:10,6:10]=2;y[7:9,7:9,7:9]=3
        image=f'{cid}.nii.gz';mask=f'{cid}_mask.nii.gz';save_nii(root/image,x);save_nii(root/mask,y)
        h=hashlib.sha256(str(x.shape).encode())
        for c in range(4):h.update(np.ascontiguousarray(x[...,c],dtype='<f4').tobytes())
        records.append({'case_id':cid,'image':image,'mask':mask,'image_voxel_sha256':h.hexdigest()})
    dump(root/'dataset.json',{'modality':MODALITIES,'labels':LABELS,'numTraining':3,'training':[{'image':r['image'],'label':r['mask']} for r in records]})
    for s,r in zip(['train','validation','internal_test'],records):
        with open(splitdir/f'{s}.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
    dump(splitdir/'split_manifest.json',{'dataset_root':str(root),'dataset_json_sha256':sha(root/'dataset.json'),'counts':{'train':1,'validation':1,'internal_test':1}})

    (root/records[-1]['image']).unlink();(root/records[-1]['mask']).unlink()
    return project

def test_synthetic_train_evaluate_and_resume_without_test_files(tmp_path):
    project=synthetic_project(tmp_path)
    args=parser().parse_args(['--project',str(project),'--output','runs/test','--epochs','1','--steps','2','--patch','16','16','16'])
    train(args);ck=project/'runs/test/best.pt';assert ck.is_file()
    args.resume=True;train(args)
    evaluate(SimpleNamespace(project=str(project),output='reports/eval',checkpoint=str(ck),manifest=None,contract=None,mc_passes=2))
    report=json.loads((project/'reports/eval/metrics.json').read_text());assert len(report['cases'])==1
    assert (project/'reports/eval/CASE_1/prediction.nii.gz').is_file()
    assert (project/'reports/eval/CASE_1/mc_mutual_information.nii.gz').is_file()

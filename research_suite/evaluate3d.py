import argparse
import time
from pathlib import Path
import numpy as np
import nibabel as nib
from common import dump, read, rows, sha, device, load_checkpoint, csv_out
from networks import UNet3D
from volumes import sliding, uncertainty_maps
from metrics import region_metrics, summarize, risk_coverage
from msd_data import load_splits, prepare_case
from engine import prepare_inputs, mask_bytes

def nifti(path, array_zyx, affine, dtype=np.float32):
    image=nib.Nifti1Image(np.asarray(array_zyx.transpose(2,1,0),dtype=dtype),affine)
    image.set_qform(affine,1); image.set_sform(affine,1); image.header.set_xyzt_units('mm'); nib.save(image,path)

def evaluate(args):
    project=Path(args.project).resolve(); out=project/args.output
    if out.exists(): raise ValueError('Use a new output folder; completed/failing evaluations are preserved')
    ck=load_checkpoint(args.checkpoint); config=ck['config']
    if config['task']!='segmentation3d': raise ValueError('Expected research-suite 3D checkpoint')
    model=UNet3D(len(config['channels'])).to(device()); model.load_state_dict(ck['model_state'])
    plan={'checkpoint_sha256':sha(args.checkpoint),'checkpoint_epoch':ck['epoch'],
        'mode':'external' if args.manifest else 'development_validation','mc_passes':args.mc_passes,
        'point_prediction':'deterministic argmax; stochastic predictions do not replace it',
        'label_mapping':{'0':'background','1':'edema','2':'non-enhancing tumor','3':'enhancing tumour'}}
    source_hashes=set()
    fingerprint_file=Path(args.checkpoint).parent/'fingerprints.json'
    if not fingerprint_file.is_file(): raise FileNotFoundError('Checkpoint companion fingerprints.json is required')
    for r in read(fingerprint_file).values(): source_hashes.update(r.values())
    if args.manifest:
        if not args.contract: raise ValueError('External evaluation needs --contract with reviewed provenance/label mapping')
        contract=read(args.contract)
        for key in ['dataset_version','license_reference','identity_evidence','overlap_review','preprocessing_evidence','label_evidence']:
            if not isinstance(contract.get(key),str) or not contract[key].strip() or contract[key].startswith('REPLACE'): raise ValueError(f'External contract lacks {key}')
        if contract.get('independence_verified') is not True: raise ValueError('External cohort independence not verified')
        mapping={int(k):int(v) for k,v in contract['label_map_to_msd'].items()}
        if mapping.get(0)!=0 or not set(mapping.values())<=set(range(4)): raise ValueError('Invalid external label map')
        records=rows(args.manifest); keys=['case_id','participant_id','flair','t1','t1ce','t2','mask']
        if not records or not all(all(r.get(k) for k in keys) for r in records): raise ValueError('External manifest missing required values')
        if len({r['participant_id'] for r in records})!=len(records): raise ValueError('Use one prespecified exam per independent participant')
        if len({r['case_id'] for r in records})!=len(records): raise ValueError('Duplicate external case IDs')
        plan.update(manifest_sha256=sha(args.manifest),contract=contract,case_ids=[r['case_id'] for r in records])
    else:
        root,splits,split_hash=load_splits(project)
        if config['split_sha256']!=split_hash: raise ValueError('Frozen splits changed')
        records=[r for r in splits['validation'] if r['case_id'] in config['case_ids']['validation']]
        plan.update(split_sha256=split_hash,case_ids=[r['case_id'] for r in records])
    out.mkdir(parents=True); dump(out/'evaluation_plan.json',plan); results=[]; flat=[]; seen=set()
    for i,r in enumerate(records,1):
        print(f'Evaluation {i}/{len(records)}: {r["case_id"]}',flush=True)
        if not r['case_id'].replace('_','').replace('-','').isalnum(): raise ValueError('Unsafe case ID')
        case_started=time.perf_counter(); native_case=None
        if args.manifest:
            for col in ['flair','t1','t1ce','t2']:
                h=sha(r[col])
                if h in source_hashes or h in seen: raise ValueError('Duplicate external/development image file')
                seen.add(h)
            native_case=prepare_inputs([r[k] for k in ['flair','t1','t1ce','t2']])
            x=native_case['images']; affine=native_case['affine']; spacing=native_case['spacing'][::-1]
            mask=nib.load(r['mask']); original_values=mask.get_fdata(dtype=np.float32)
            if not np.isfinite(original_values).all() or not np.allclose(original_values,np.rint(original_values)): raise ValueError('Invalid reference values')
            mask=nib.as_closest_canonical(mask); ref=mask.get_fdata(dtype=np.float32)
            if ref.shape!=tuple(native_case['shape']) or not np.allclose(mask.affine,affine,atol=1e-3): raise ValueError('Reference grid differs from MRI')
            if not set(np.unique(ref)).issubset(mapping): raise ValueError('Unmapped external mask label')
            y=np.zeros(ref.shape,np.uint8)
            for old,new in mapping.items(): y[ref==old]=new
            y=y.transpose(2,1,0)
            if not np.allclose(spacing,[1,1,1],atol=1e-3): raise ValueError('External input requires reviewed 1-mm preprocessing for this model')
        else:
            info=prepare_case(root,r,project/'data/stage3_cache_v1'); folder=project/'data/stage3_cache_v1'/r['case_id']
            x=np.load(folder/'images.npy',mmap_mode='r'); y=np.load(folder/'masks.npy',mmap_mode='r')
            affine=np.array(info['canonical_affine']); spacing=info['spacing'][::-1]
        prob=sliding(model,x,device(),config['patch'],config['channels']); pred=prob.argmax(0).astype(np.uint8)
        entropy=-(prob*np.log(np.maximum(prob,1e-8))).sum(0)/np.log(4)
        fg=np.any(x!=0,axis=1); tumor=pred>0; region=region_metrics(pred,y,spacing)
        case={'case_id':r['case_id'],'regions':region,
            'mean_brain_entropy':float(entropy[fg].mean()) if fg.any() else None,
            'mean_predicted_tumor_entropy':float(entropy[tumor].mean()) if tumor.any() else None}
        folder=out/r['case_id']; folder.mkdir()
        nifti(folder/'prediction.nii.gz',pred,affine,np.uint8)
        nifti(folder/'entropy.nii.gz',entropy,affine)

        errors=np.zeros_like(pred); errors[(pred>0)&(y==0)]=1; errors[(pred==0)&(y>0)]=2
        errors[(pred>0)&(y>0)&(pred!=y)]=3; nifti(folder/'errors.nii.gz',errors,affine,np.uint8)
        if native_case: (folder/'prediction_native.nii.gz').write_bytes(mask_bytes(pred.transpose(2,1,0),native_case))
        if args.mc_passes:
            if args.mc_passes<2: raise ValueError('MC dropout requires >=2 passes')
            _,mc_entropy,mi=uncertainty_maps(model,x,device(),config['patch'],config['channels'],args.mc_passes)
            nifti(folder/'mc_predictive_entropy.nii.gz',mc_entropy,affine); nifti(folder/'mc_mutual_information.nii.gz',mi,affine)
            case['mean_brain_mc_mutual_information']=float(mi[fg].mean()) if fg.any() else None
        case['inference_and_metrics_seconds']=time.perf_counter()-case_started
        dump(folder/'metrics.json',case); results.append(case)
        for name,m in region.items(): flat.append({'case_id':r['case_id'],'region':name,**m})
    report={'summary':summarize(results),'cases':results,'evaluation':plan['mode'],
        'note':'Conditional case-level intervals; not patient-independent unless provenance establishes it. No model fitting or threshold selection.'}
    for key in ['mean_brain_entropy','mean_predicted_tumor_entropy','mean_brain_mc_mutual_information']:
        selected=[r for r in results if r.get(key) is not None and r['regions']['whole_tumor']['dice'] is not None]
        report[key+'_risk_coverage']=risk_coverage([1-r['regions']['whole_tumor']['dice'] for r in selected],[r[key] for r in selected])
        report[key+'_risk_coverage']['included_cases']=len(selected)
    dump(out/'metrics.json',report); csv_out(out/'per_region.csv',flat); print(report['summary'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Full-volume validation or explicitly reviewed external evaluation.'); p.add_argument('--project',default='.')
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--manifest'); p.add_argument('--contract'); p.add_argument('--mc-passes',type=int,default=0)
    evaluate(p.parse_args())

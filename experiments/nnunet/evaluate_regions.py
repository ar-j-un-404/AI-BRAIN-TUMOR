import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import nibabel as nib

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.project/'research_suite'))
    from metrics import region_metrics,summarize
    from msd_data import load_splits
    root,splits,hashes=load_splits(a.project)
    expected={r['case_id'] for r in splits['validation']}
    actual={f.name[:-7] for f in a.predictions.glob('*.nii.gz')}
    if len(expected)!=73 or actual!=expected:raise ValueError(f'Prediction case mismatch. Missing: {expected-actual}; Extra: {actual-expected}')
    cases=[]
    for i,r in enumerate(splits['validation'],1):
        print(f'Region evaluation {i}/73: {r["case_id"]}',flush=True)
        pred=nib.as_closest_canonical(nib.load(a.predictions/(r['case_id']+'.nii.gz')))
        ref=nib.as_closest_canonical(nib.load(root/r['mask']))
        if pred.shape!=ref.shape or not np.allclose(pred.affine,ref.affine,atol=1e-3):raise ValueError('Prediction/reference grid mismatch')
        arrays=[]
        for image in (pred,ref):
            values=image.get_fdata(dtype=np.float32)
            if not np.isfinite(values).all() or not np.allclose(values,np.rint(values)) or not set(np.unique(values))<=set(range(4)):raise ValueError('Invalid segmentation labels')
            arrays.append(values.astype(np.uint8))
        cases.append({'case_id':r['case_id'],'regions':region_metrics(*arrays,spacing=ref.header.get_zooms()[:3])})
    result={'evaluation':'development_validation','summary':summarize(cases),'cases':cases,
        'split_sha256':hashes,'note':'nnU-Net 250-epoch, 2-GB-plan, custom single-fold development benchmark. Final checkpoint, no postprocessing selection. Planning used train+validation. Not independent test or equal-compute architecture comparison.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()

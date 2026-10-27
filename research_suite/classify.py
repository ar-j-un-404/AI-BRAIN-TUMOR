import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar
from scipy.special import expit
from common import rows, sha, dump, read, seed_all, device, save_checkpoint, load_checkpoint, csv_out, code_hashes
from networks import Classifier3D
from metrics import classification_metrics, risk_coverage
from msd_data import check_header

SPLITS={'train','validation','calibration','test','external'}
REQUIRED={'case_id','participant_id','source','split','label','t1','t2','identity_evidence','protocol_evidence','label_evidence'}

def validate_manifest(path, partitions=('train','validation','calibration'), inspect_files=True):
    records=rows(path)
    if not records or not REQUIRED<=set(records[0]): raise ValueError('Manifest is empty or lacks required columns; see templates/classification_manifest.csv')
    cases=set(); groups={}; by_source=defaultdict(set)
    for r in records:
        if r['case_id'] in cases: raise ValueError('Duplicate case ID')
        cases.add(r['case_id'])
        if not all(r[k].strip() for k in REQUIRED): raise ValueError('Missing evidence/metadata')
        if r['split'] not in SPLITS or r['label'] not in {'0','1'}: raise ValueError('Invalid split or label')

        key=r['participant_id']; signature=(r['split'],r['label'])
        if key in groups and groups[key]!=signature: raise ValueError('Participant overlaps splits or has conflicting labels')
        groups[key]=signature
        if r['split'] in partitions: by_source[r['source']].add(r['label'])
    for s in partitions:
        subset=[r for r in records if r['split']==s]
        if {r['label'] for r in subset}!={'0','1'}: raise ValueError(f'{s} must contain both classes')

    confounded=[s for s,labels in by_source.items() if len(labels)<2]
    if confounded: raise ValueError(f'Sources contain only one class: {confounded}. Obtain within-source controls/tumors; do not rename sources to bypass this check.')
    if inspect_files:
        hashes={}; volumes={}
        for r in records:
            if r['split'] not in partitions: continue
            for col in ('t1','t2'):
                file=Path(r[col])
                if not file.is_file(): raise FileNotFoundError(str(file))
                digest=sha(file)
                if digest in hashes: raise ValueError(f'Duplicate file: {r["case_id"]} and {hashes[digest]}')
                hashes[digest]=r['case_id']
            a, geometry=load_pair(r, return_geometry=True)
            digest=__import__('hashlib').sha256(a.numpy().tobytes()).hexdigest()
            if digest in volumes: raise ValueError(f'Duplicate standardized volume: {r["case_id"]} and {volumes[digest]}')
            volumes[digest]=r['case_id']
    return records

def load_pair(row, return_geometry=False):
    volumes=[]; affine=None; spatial_shape=None
    for name in ('t1','t2'):
        img=nib.load(row[name]); check_header(img)
        if len(img.shape)!=3 or np.prod(img.shape)>30_000_000: raise ValueError('Expected bounded 3D T1/T2 volumes')
        if img.header.get_xyzt_units()[0]!='mm': raise ValueError('Millimetre units required')
        img=nib.as_closest_canonical(img); spacing=nib.affines.voxel_sizes(img.affine)
        if not np.allclose(img.affine[:3,:3]/spacing,np.eye(3),atol=1e-3): raise ValueError('Oblique/sheared input: preprocess before use')
        if affine is not None and (img.shape!=spatial_shape or not np.allclose(img.affine,affine,atol=1e-3)): raise ValueError('T1/T2 must already be registered to the same grid')
        affine=img.affine; spatial_shape=img.shape
        x=img.get_fdata(dtype=np.float32)
        if not np.isfinite(x).all(): raise ValueError('Non-finite input')
        mask=x!=0
        if not mask.any() or x[mask].std()<1e-8: raise ValueError('Empty/constant input')
        out=np.zeros_like(x); out[mask]=(x[mask]-x[mask].mean(dtype=np.float64))/x[mask].std(dtype=np.float64)
        volumes.append(np.clip(out,-5,5).transpose(2,1,0))
    x=np.stack(volumes); foreground=np.any(x!=0,axis=0)
    bounds=[(int(v.min()),int(v.max())+1) for v in np.where(foreground)]
    crop=tuple(slice(a,b) for a,b in bounds); x=x[(slice(None),)+crop]

    tensor=F.interpolate(torch.from_numpy(np.ascontiguousarray(x))[None],size=(64,64,64),mode='trilinear',align_corners=False)[0]
    if return_geometry: return tensor, {'canonical_affine':affine.tolist(),'canonical_shape':list(spatial_shape),'crop_zyx':bounds}
    return tensor

def features(tensor):
    x=tensor.numpy(); values=[]
    for v in x:
        fg=v[v!=0]
        values.extend([float(fg.mean()),float(fg.std()),*np.percentile(fg,[5,25,50,75,95]).tolist(),float((v!=0).mean())])
    return values

def aggregate(records, logits):
    groups=defaultdict(list); labels={}; sources={}
    for r,z in zip(records,logits):
        key=r['participant_id']; groups[key].append(float(z)); labels[key]=int(r['label']); sources[key]=r['source']
    keys=sorted(groups)
    return keys,np.array([labels[k] for k in keys]),np.array([np.mean(groups[k]) for k in keys]),[sources[k] for k in keys]

def bootstrap_classification(y, probabilities, threshold=.5):
    rng=np.random.default_rng(42); values=defaultdict(list)
    for _ in range(1000):

        idx=np.concatenate([rng.choice(np.flatnonzero(y==c),sum(y==c),replace=True) for c in (0,1)])
        m=classification_metrics(y[idx],probabilities[idx],threshold)
        for k in ('accuracy','balanced_accuracy','recall','specificity','roc_auc','f1','pr_auc_average_precision','brier'):
            if m[k] is not None: values[k].append(m[k])
    return {k:np.percentile(v,[2.5,97.5]).tolist() for k,v in values.items()}

def subject_report(records, logits, temperature=1., threshold=.5):
    ids,y,z,sources=aggregate(records,logits); p=expit(z/temperature)
    report=classification_metrics(y,p,threshold)
    report['conditional_stratified_participant_bootstrap_95ci']=bootstrap_classification(y,p,threshold)
    entropy=-(p*np.log(np.maximum(p,1e-8))+(1-p)*np.log(np.maximum(1-p,1e-8)))/np.log(2)
    report['risk_coverage']=risk_coverage(((p>=threshold)!=y).astype(float),entropy)
    report['by_source']={s:classification_metrics(y[np.array(sources)==s],p[np.array(sources)==s],threshold) for s in sorted(set(sources))}
    report['participants']=len(ids); report['recordings']=len(records)
    return report,[{'participant_id':i,'label':int(t),'score':float(q),'source':s} for i,t,q,s in zip(ids,y,p,sources)]

def train(args):
    manifest=Path(args.manifest).resolve(); out=Path(args.output)
    if out.exists(): raise ValueError('Output exists. Use a new run name.')
    print('Verifying identity, source balance and training/development files...',flush=True)
    records=validate_manifest(manifest); out.mkdir(parents=True)
    if not args.channels or len(set(args.channels))!=len(args.channels) or not set(args.channels)<={0,1}: raise ValueError('Classifier channels must be 0 (T1), 1 (T2), or both')
    groups={s:[r for r in records if r['split']==s] for s in ('train','validation','calibration')}
    seed_all(args.seed); dev=device(); tensors={}
    for s,rs in groups.items():
        for i,r in enumerate(rs,1):
            print(f'Loading {s} {i}/{len(rs)}',flush=True); tensors[r['case_id']]=load_pair(r)[args.channels]
    config={'task':'classification3d','protocol':'registered-skullstripped-T1-T2-foreground-crop64-v1',
        'channels':[ ['T1w','T2w'][c] for c in args.channels ],'input_indices':args.channels,'manifest_sha256':sha(manifest),'seed':args.seed,'epochs':args.epochs,
        'code_sha256':code_hashes(),'torch_version':str(torch.__version__),
        'participants':{s:sorted({r['participant_id'] for r in rs}) for s,rs in groups.items()}}
    dump(out/'config.json',config)
    fingerprints={r['case_id']:{col:sha(r[col]) for col in ('t1','t2')} for s in groups for r in groups[s]}
    dump(out/'fingerprints.json',fingerprints)

    X=np.array([features(tensors[r['case_id']]) for r in groups['train']]); y=np.array([int(r['label']) for r in groups['train']])
    baseline=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=args.seed))
    participant_counts=__import__('collections').Counter(r['participant_id'] for r in groups['train'])
    weights=np.array([1/participant_counts[r['participant_id']] for r in groups['train']])
    baseline.fit(X,y,logisticregression__sample_weight=weights); joblib.dump(baseline,out/'baseline.joblib')
    val=groups['validation']; val_X=np.array([features(tensors[r['case_id']]) for r in val])
    report,_=subject_report(val,baseline.decision_function(val_X)); dump(out/'baseline_validation.json',report)
    df=pd.DataFrame(groups['train']); vdf=pd.DataFrame(val)
    for name,numeric,categorical in [('source_only',[],['source']),('demographics',['age'],['sex'])]:
        if not all(c in df and c in vdf for c in numeric+categorical): continue
        if numeric:
            for col in numeric: df[col]=pd.to_numeric(df[col],errors='coerce'); vdf[col]=pd.to_numeric(vdf[col],errors='coerce')
            if df[numeric].isna().all().any(): continue
        prep=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric),
            ('categorical',OneHotEncoder(handle_unknown='ignore'),categorical)])
        probe=make_pipeline(prep,LogisticRegression(class_weight='balanced',max_iter=2000))
        probe.fit(df,y,logisticregression__sample_weight=weights)
        report,_=subject_report(val,probe.decision_function(vdf)); dump(out/f'{name}_validation.json',report)
    model=Classifier3D(len(args.channels)).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda'); best=-1.; history=[]

    labels={r['participant_id']:int(r['label']) for r in groups['train']}
    pos=sum(labels.values()); neg=len(labels)-pos; pos_weight=torch.tensor([neg/pos],device=dev)
    tr=groups['train']; rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for index in rng.permutation(len(tr)):
            r=tr[index]; x=tensors[r['case_id']].clone()
            if rng.random()<.5: x=x.flip(-1)
            x=x[None].to(dev); target=torch.tensor([int(r['label'])],dtype=torch.float32,device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda',dtype=torch.float16):
                loss=F.binary_cross_entropy_with_logits(model(x),target,pos_weight=pos_weight)/participant_counts[r['participant_id']]
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode(): logits=[float(model(tensors[r['case_id']][None].to(dev))[0]) for r in val]
        _,vy,vz,_=aggregate(val,logits); score=classification_metrics(vy,expit(vz))['balanced_accuracy']
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_participant_balanced_accuracy':score})
        print(f'Classifier epoch {epoch}/{args.epochs}: validation balanced accuracy {score:.4f}',flush=True)
        if score>best:
            best=score; save_checkpoint(out/'best.pt',{'model_state':model.state_dict(),'config':config,'epoch':epoch,'temperature':1.,'threshold':.5})
    dump(out/'history.json',history); saved=load_checkpoint(out/'best.pt'); model.load_state_dict(saved['model_state']); model.eval()
    cal=groups['calibration']
    with torch.inference_mode(): logits=[float(model(tensors[r['case_id']][None].to(dev))[0]) for r in cal]
    _,cy,cz,_=aggregate(cal,logits)
    def nll(logt):
        z=cz/np.exp(logt); return float(np.mean(np.logaddexp(0,z)-cy*z))
    solution=minimize_scalar(nll,bounds=(-3,3),method='bounded')
    saved['temperature']=float(np.exp(solution.x)); saved['threshold']=.5
    saved['calibration_note']='Temperature fitted on separate calibration participants. Threshold fixed at 0.5; not clinically validated.'
    save_checkpoint(out/'classifier.pt',saved)
    report,predictions=subject_report(cal,logits,saved['temperature']); report['evaluation']='calibration fit diagnostics, not held-out performance'
    dump(out/'calibration_fit.json',report); csv_out(out/'calibration_scores.csv',predictions)
    with torch.inference_mode(): logits=[float(model(tensors[r['case_id']][None].to(dev))[0]) for r in val]
    report,predictions=subject_report(val,logits,saved['temperature']); report['evaluation']='checkpoint-selection development validation'
    dump(out/'validation.json',report); csv_out(out/'validation_scores.csv',predictions)
    print('Classifier trained. Reserved test/external scans were not opened. Run evaluate only after freezing this model.')

def evaluate(args):
    out=Path(args.output)
    if out.exists(): raise ValueError('Evaluation output already exists; retain the original result')
    ck=load_checkpoint(args.checkpoint); records=validate_manifest(args.manifest,partitions=(),inspect_files=False)
    if sha(args.manifest)!=ck['config']['manifest_sha256']: raise ValueError('Frozen manifest changed; use the originally declared test/external rows')
    selected=[r for r in records if r['split']==args.split]
    if {r['label'] for r in selected}!={'0','1'}: raise ValueError('Evaluation requires both classes')
    out.mkdir(parents=True)
    dump(out/'freeze.json',{'checkpoint_sha256':sha(args.checkpoint),'manifest_sha256':sha(args.manifest),'split':args.split,
        'threshold':ck['threshold'],'temperature':ck['temperature'],'case_ids':[r['case_id'] for r in selected]})
    dev=device(); model=Classifier3D(len(ck['config']['input_indices'])).to(dev); model.load_state_dict(ck['model_state']); model.eval(); logits=[]
    training_fingerprints=read(Path(args.checkpoint).parent/'fingerprints.json')
    existing={v for r in training_fingerprints.values() for v in r.values()}
    for r in selected:
        for col in ('t1','t2'):
            if sha(r[col]) in existing: raise ValueError('Evaluation file duplicates development file')
        with torch.inference_mode(): logits.append(float(model(load_pair(r)[ck['config']['input_indices']][None].to(dev))[0]))
    report,predictions=subject_report(selected,logits,ck['temperature'],ck['threshold'])
    report['evaluation']=args.split; dump(out/'metrics.json',report); csv_out(out/'scores.csv',predictions)
    (out/'metrics.txt').write_text(str(report),encoding='utf-8'); print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Separate T1/T2 tumor-presence experiment, gated by verified participant metadata.'); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('train'); a.add_argument('--manifest',required=True); a.add_argument('--output',default='runs/classification_001')
    a.add_argument('--channels',type=int,nargs='+',default=[0,1]); a.add_argument('--epochs',type=int,default=20); a.add_argument('--seed',type=int,default=42)
    b=sub.add_parser('evaluate'); b.add_argument('--manifest',required=True); b.add_argument('--checkpoint',required=True)
    b.add_argument('--split',choices=['test','external'],required=True); b.add_argument('--output',required=True)
    args=p.parse_args(); train(args) if args.command=='train' else evaluate(args)

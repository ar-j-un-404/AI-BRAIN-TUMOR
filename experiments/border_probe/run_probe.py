import argparse
import csv
import hashlib
import io
import json
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
from features import features,preview
from protocol import load_records,metrics

BASE=Path(__file__).resolve().parent

def dump(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def validate_source(root,expected):
    config=json.loads((root/'experiment.json').read_text())
    if config['version']!='btc-ixi-physical-paired-v2':
        raise ValueError('Select results from BrainTumor_Physical_Space_v2, not another experiment.')
    records=config['records']
    fields=['id','source','site','label','fold','expected_hash']
    def signature(rows):return sorted(tuple(r[k] for k in fields) for r in rows)
    if signature(records)!=signature(expected):raise ValueError('Participant identities, hashes or folds changed')
    if len({r['id'] for r in records})!=500:raise ValueError('Duplicate participant IDs')
    return records

def load_cache(root,r,physical):
    cache=root/'cache';meta=json.loads((cache/(r['id']+'.json')).read_text())
    if meta['input']['expected_hash']!=r['expected_hash']:raise ValueError('Cache input identity mismatch: '+r['id'])
    name=r['id']+('_physical' if physical else '')+'.npy'
    raw=(cache/name).read_bytes()
    key='physical_sha256' if physical else 'cache_sha256'
    if hashlib.sha256(raw).hexdigest()!=meta[key]:raise ValueError('Cache SHA256 mismatch: '+name)
    bag=np.load(io.BytesIO(raw),allow_pickle=False)
    if bag.shape!=(24,1,96,96) or not np.isfinite(bag).all():raise ValueError('Invalid cache: '+name)
    return bag

def fit_predict(X,y,folds,records,task):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    rows=[];checks=[]
    for fold in range(1,6):
        tr=folds!=fold;te=folds==fold
        assert set(np.array([r['id'] for r in records])[tr]).isdisjoint(np.array([r['id'] for r in records])[te])
        if len(np.unique(y[tr]))!=2 or len(np.unique(y[te]))!=2:
            raise ValueError('Each diagnostic fold must contain both classes')
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=42))

        weights=np.ones(int(tr.sum()))
        training=[r for r,t in zip(records,tr) if t]
        if task=='pooled_disease':
            pools=[('tumor',[i for i,r in enumerate(training) if r['label']==1],.5),
                   ('BTC_healthy',[i for i,r in enumerate(training) if r['source']=='BTC' and r['label']==0],.25)]
            sites=sorted({r['site'] for r in training if r['source']=='IXI'})
            pools += [(s,[i for i,r in enumerate(training) if r['source']=='IXI' and r['site']==s],.25/len(sites)) for s in sites]
            for _,idx,mass in pools:
                if not idx:raise ValueError('Empty training pool')
                weights[idx]=mass*len(training)/len(idx)
            model[-1].set_params(class_weight=None)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(X[tr],y[tr],logisticregression__sample_weight=weights)
        scores=model.predict_proba(X[te])[:,1]
        for i,score in zip(np.flatnonzero(te),scores):
            r=records[i]
            rows.append(dict(id=r['id'],source=r['source'],site=r['site'],label=int(y[i]),fold=fold,score=float(score)))
        checks.append(dict(fold=fold,train_ids=[r['id'] for r,t in zip(records,tr) if t],
                           evaluation_ids=[r['id'] for r,t in zip(records,te) if t]))
    return rows,checks

def run(root,out):
    import sklearn
    records=validate_source(root,load_records(BASE))
    dump(out/'protocol.json',dict(version='border-statistics-v1',cache_source=str(root),
        records=[{k:r[k] for k in ['id','source','site','label','fold','expected_hash']} for r in records],
        masked_center='rows and columns 12:84; same across all 24 slices',
        model='StandardScaler + LogisticRegression C=1 liblinear fixed threshold .5',
        features='168 summary features from fixed spatial/slice bands; no selection or tuning',
        reserved_accessed=False,numpy=np.__version__,sklearn=sklearn.__version__))
    results={};predictions=[];fold_checks={};lines=[]
    for physical in [False,True]:
        protocol='physical_320mm' if physical else 'voxel_cube'
        print('Extracting cached features:',protocol,flush=True)
        values={'full':[],'border':[]}
        for i,r in enumerate(records,1):
            bag=load_cache(root,r,physical)
            for kind in values:values[kind].append(features(bag,kind))

            if (r['source']=='BTC' and r['label']==0) or r['id'] in ['sub-PAT01','sub-PAT02','IXI002','IXI012']:
                preview(bag,out/(protocol+'_'+r['id']+'.png'),protocol+' '+r['id'])
            if i%100==0:print(f'  {i}/500',flush=True)
        for kind,X in values.items():
            X=np.array(X)
            for task in ['BTC_disease','pooled_disease','healthy_source']:
                keep=np.array([r['source']=='BTC' if task=='BTC_disease' else r['label']==0 if task=='healthy_source' else True for r in records])
                selected=[r for r,t in zip(records,keep) if t]
                y=np.array([int(r['source']=='BTC') if task=='healthy_source' else r['label'] for r in selected])
                folds=np.array([r['fold'] for r in selected])
                name=protocol+' / '+kind+' / '+task
                print('Fitting:',name,flush=True)
                rows,checks=fit_predict(X[keep],y,folds,selected,task)
                fold_checks[name]=checks
                groups={'all':rows} if task!='pooled_disease' else {g:[r for r in rows if r['source']==g or r['site']==g] for g in ['BTC','IXI','Guys','HH','IOP']}
                results[name]={}
                for group,subset in groups.items():
                    m=metrics(subset);results[name][group]=m
                    desc=name+' / '+group+' | n='+str(m['n'])
                    for key in ['accuracy','balanced_accuracy','sensitivity','specificity']:
                        if key in m and m[key] is not None:desc+=f' | {key} {100*m[key]:.2f}%'
                    lines.append(desc+' | CM '+str(m['confusion_matrix']))
                predictions.extend([dict(experiment=name,**r) for r in rows])
    dump(out/'results.json',dict(results=results,interpretation='Exploratory border probe. Source prediction does not prove the CNN uses these cues. Borders may contain anatomy; cached normalization uses whole-image intensities. Healthy-source positive class is BTC, not tumor.'))
    dump(out/'folds.json',fold_checks)
    (out/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    print('\n'+'\n'.join(lines),flush=True)
    print('Complete. No app, MRI, cache or checkpoint changed. No deployment model saved.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
    out=BASE/'reports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    try:run(args.results,out)
    except Exception:
        text=traceback.format_exc();(out/'error.txt').write_text(text,encoding='utf-8');print(text);raise
    finally:
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():
                if p.is_file():z.write(p,p.name)
            for name in ['features.py','run_probe.py','protocol.py']:z.write(BASE/name,'code/'+name)
        print('UPLOAD THIS ZIP:',archive,flush=True)

import csv
import hashlib
import json
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from protocol import metrics
from sampling import make_schedule

BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
STEPS=1000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
def dump(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def read_inputs(root,ref):
    actual=json.loads((root/'results/experiment.json').read_text())
    if digest(actual)!=ref['source_experiment_hash']:raise ValueError('Comparison experiment differs from reviewed report')
    if sha(BASE/'core.py')!=actual['source_hashes']['core.py']:raise ValueError('Model code changed')
    records=actual['records'];btc=[r for r in records if r['source']=='BTC']
    if len(records)!=500 or len({r['id'] for r in records})!=500 or len(btc)!=36:raise ValueError('Unexpected participant manifest')
    bags={}
    for r in btc:
        p=root/'results/cache'/(r['id']+'_brain.npy')
        if sha(p)!=ref['brain_cache'][r['id']]:raise ValueError('Brain cache changed: '+r['id'])
        a=np.load(p,allow_pickle=False)
        if a.shape!=(24,1,96,96) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid cached input')
        bags[r['id']]=a
    return records,btc,bags

def run():
    import torch
    from core import Classifier,seed_all
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use the main working environment')
    root=Path(json.loads((BASE/'settings.json').read_text())['comparison_folder'])
    ref=json.loads((BASE/'reference.json').read_text())
    records,btc,arrays=read_inputs(root,ref)
    pooled=ref['pooled_predictions']
    if len(pooled)!=36 or {r['id'] for r in pooled}!={r['id'] for r in btc}:raise ValueError('Invalid pooled reference coverage')
    byid={r['id']:r for r in records}
    for r in pooled:
        if r['label']!=byid[r['id']]['label'] or r['fold']!=byid[r['id']]['fold']:raise ValueError('Reference labels/folds mismatch')
    config=dict(version='btc-only-extracted-paired-v1',steps=1000,threshold=.5,seed=42,lr=.001,
                training='2 BTC tumors +2 distinct BTC controls per update; no IXI images',
                source_experiment=ref['source_experiment_hash'],brain_cache=ref['brain_cache'],
                sources={n:sha(BASE/n) for n in ['run_training.py','sampling.py','core.py','protocol.py','reference.json']})
    identity=digest(config)
    cfg=OUT/'experiment.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Experiment changed; preserve results before starting a different experiment')
    dump(cfg,config)
    (OUT/'error.txt').unlink(missing_ok=True)
    bags={i:torch.from_numpy(a)[None] for i,a in arrays.items()}
    device=torch.device('cuda');new=[];fitting=[]
    dump(OUT/'environment.json',dict(torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(0)))
    for fold in range(1,6):
        train=[r for r in btc if r['fold']!=fold];test=[r for r in btc if r['fold']==fold]
        train_ids={r['id'] for r in train};test_ids={r['id'] for r in test}
        if train_ids&test_ids:raise ValueError('Participant overlap')
        schedule=make_schedule(records,fold,STEPS)
        if any(i not in train_ids for s in schedule for i in s['btc_ids']):raise ValueError('Nontraining participant sampled')
        folder=OUT/f'fold_{fold}';folder.mkdir(exist_ok=True)
        dump(folder/'checkpoint_protocol.json',dict(train_ids=sorted(train_ids),evaluation_ids=sorted(test_ids),steps=STEPS,
             sampling_schedule_sha256=digest(schedule),note='Final fixed step. No held-out selection; prior masks unverified.'))
        seed_all(42);model=Classifier().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        objective=torch.nn.BCEWithLogitsLoss();start=0;exposures=Counter();checkpoint=folder/'resume.pt'
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location=device,weights_only=True)
            if state.get('identity')!=dict(experiment=identity,fold=fold):raise ValueError('Wrong checkpoint')
            start=state['step']
            if not isinstance(start,int) or not 0<=start<=STEPS:raise ValueError('Invalid resume step')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);exposures.update(state['exposures'])
            del state
        print(f'BTC ONLY / BRAIN EXTRACTED | fold {fold}/5 | starting step {start}',flush=True)
        tick=time.monotonic()
        for step in range(start+1,STEPS+1):
            model.train();optimizer.zero_grad(set_to_none=True);total=0.
            for i in schedule[step-1]['btc_ids']:
                loss=objective(model(bags[i].to(device)),torch.tensor([byid[i]['label']],dtype=torch.float32,device=device))/4
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();total+=float(loss.detach());exposures[i]+=1
            optimizer.step()
            if step%100==0:
                print(f'Step {step}/{STEPS} | batch loss {total:.4f} | {time.monotonic()-tick:.1f}s per100steps',flush=True);tick=time.monotonic()
                tmp=checkpoint.with_suffix('.tmp')
                torch.save(dict(identity=dict(experiment=identity,fold=fold),model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,exposures=dict(exposures)),tmp);tmp.replace(checkpoint)
        dump(folder/'sampling.json',dict(exposures))
        model.eval();fold_rows=[]
        with torch.inference_mode():
            for role,group in [('training_subset',train),('held_out',test)]:
                rows=[]
                for r in group:
                    score=float(torch.sigmoid(model(bags[r['id']].to(device))).item())
                    if not np.isfinite(score):raise ValueError('Nonfinite prediction')
                    row=dict(id=r['id'],label=r['label'],fold=fold,score=score,prediction=int(score>=.5),role=role)
                    rows.append(row);fold_rows.append(row)
                m=metrics(rows);print(role,json.dumps(m),flush=True)
                fitting.append(dict(fold=fold,role=role,**m))
                if role=='held_out':new+=rows
        dump(folder/'predictions.json',fold_rows)
        del model,optimizer
        torch.cuda.empty_cache()
    if len(new)!=36 or len({r['id'] for r in new})!=36:raise ValueError('Incomplete out-of-fold results')
    before={r['id']:r for r in pooled};changes=[]
    for r in new:
        old=before[r['id']];was=old['prediction']==old['label'];now=r['prediction']==r['label']
        changes.append(dict(id=r['id'],label=r['label'],fold=r['fold'],pooled_score=old['score'],btc_only_score=r['score'],
                            change='fix' if not was and now else 'break' if was and not now else 'both_correct' if now else 'both_wrong'))
    old_metrics=metrics(pooled);new_metrics=metrics(new)
    report=dict(evaluation='Exploratory reused participant folds; same brain-extracted inputs; no untouched external validation',
        mask_quality='unverified',reserved_test_accessed=False,ixi_images_read=False,participants=36,
        results={'BTC_plus_IXI_existing':old_metrics,'BTC_only_new':new_metrics},
        paired_changes=dict(Counter(r['change'] for r in changes)),
        change_percentage_points={k:100*(new_metrics[k]-old_metrics[k]) for k in ['accuracy','balanced_accuracy','sensitivity','specificity']},
        limitation='Replacing IXI controls with BTC controls also increases BTC-control exposure. This tests that training policy, not the isolated causal effect of dataset identity.')
    dump(OUT/'comparison.json',report);dump(OUT/'paired_cases.json',changes);dump(OUT/'fold_summaries.json',fitting)
    dump(OUT/'reference_predictions.json',pooled)
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(new[0]));w.writeheader();w.writerows(new)
    lines=['BRAIN-EXTRACTED BTC: SAME36HELD-OUT PARTICIPANTS']
    for name,m in report['results'].items():
        lines.append(name)
        for k in ['accuracy','balanced_accuracy','sensitivity','specificity','precision','f1']:lines.append(f'  {k}: {100*m[k]:.2f}%')
        lines.append('  confusion matrix: '+str(m['confusion_matrix']))
    lines.append('BTC-only minus pooled (percentage points): '+json.dumps(report['change_percentage_points']))
    lines.append('Exploratory comparison only. No new model installed in the app.')
    text='\n'.join(lines);(OUT/'comparison.txt').write_text(text);print(text,flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:run()
    except Exception:
        error=traceback.format_exc();print(error,flush=True);(OUT/'error.txt').write_text(error)
    finally:
        archive=BASE/'btc_only_comparison_report.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.csv','.txt']:z.write(p,str(p.relative_to(OUT)))
            for n in ['run_training.py','sampling.py','core.py','protocol.py','reference.json','README.txt']:z.write(BASE/n,'code/'+n)
        print('UPLOAD THIS ZIP:',archive,flush=True)

import csv
import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
from collections import Counter
from protocol import load_records, metrics

BASE = Path(__file__).resolve().parent
OUT = BASE/'results'
STEPS = 1000
VARIANTS = ('voxel_cube', 'brain_extracted')


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def run():
    import numpy as np
    import torch
    from core import Classifier, seed_all
    from prepare import prepare_all, PROTOCOL
    torch.set_num_threads(2)
    records = load_records(BASE)
    if (OUT/'error.txt').exists():
        (OUT/'error.txt').unlink()
    config = dict(version='btc-ixi-extraction-paired-v1', steps=STEPS, seed=42, lr=.001,
                  threshold=.5, precision='float32', microbatch=1, accumulation=4,
                  preprocessing={'voxel_cube':'ras-zscore-clip5-cube96-24axial-mean-embedding','brain_extracted':PROTOCOL},
                  sampling='2 BTC tumor + 1 BTC healthy + 1 IXI healthy; IXI site uniform',
                  records=records, source_hashes={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()
                    for n in ('core.py','prepare.py','qc.py','protocol.py','run_training.py','btc_plan.json','ixi_development.csv','ixi_split_lock.json','extraction_source_lock.json','prior_review.json')})
    OUT.mkdir(exist_ok=True)
    config_file = OUT/'experiment.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError('Experiment changed. Move the existing results folder to preserve it before starting a different run.')
    else:
        write_json(config_file, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| 36 BTC + 464 IXI development participants', flush=True)
    print('Five folds, two variants, 1000 steps each. Reserved IXI 117: NOT USED.', flush=True)
    write_json(OUT/'environment.json', dict(torch=str(torch.__version__), numpy=np.__version__, device=str(device)))
    cache = prepare_all(BASE, OUT, records)
    cache_identity={r['id']:json.loads((cache/(r['id']+'.json')).read_text()) for r in records}
    cache_lock=OUT/'training_cache_lock.json'
    if cache_lock.exists() and json.loads(cache_lock.read_text())!=cache_identity:
        raise ValueError('Preprocessed inputs changed after training protocol was frozen. Existing checkpoints cannot be resumed with changed inputs.')
    write_json(cache_lock,cache_identity)
    experiment_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

    bags = {variant:{r['id']:torch.from_numpy(np.load(cache/(r['id']+('_brain' if variant=='brain_extracted' else '')+'.npy'),allow_pickle=False)).unsqueeze(0) for r in records} for variant in VARIANTS}
    if any(tuple(x.shape)!=(1,24,1,96,96) for group in bags.values() for x in group.values()):
        raise ValueError('Unexpected cached bag shape')
    all_predictions=[]
    for fold in range(1,6):
        train=[r for r in records if r['fold']!=fold]
        test=[r for r in records if r['fold']==fold]
        healthy=[r for r in train if r['source']=='BTC' and r['label']==0]
        tumor=[r for r in train if r['source']=='BTC' and r['label']==1]
        sites=sorted({r['site'] for r in train if r['source']=='IXI'})
        pools={s:[r for r in train if r['source']=='IXI' and r['site']==s] for s in sites}
        for variant in VARIANTS:
            folder=OUT/variant/f'fold_{fold}'; folder.mkdir(parents=True,exist_ok=True)
            seed_all(42)
            model=Classifier().to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            objective=torch.nn.BCEWithLogitsLoss()
            rng=np.random.default_rng(42)
            start=0; exposures=Counter()
            state_file=folder/'resume.pt'
            if state_file.exists():
                state=torch.load(state_file,map_location=device,weights_only=True)
                if state.get('identity')!=dict(experiment=experiment_hash,fold=fold,variant=variant):raise ValueError('Checkpoint belongs to a different experiment/fold/variant')
                model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                rng.bit_generator.state=json.loads(state['rng_json'])
                start=state['step'];exposures.update(state['exposures'])
                if not isinstance(start,int) or not 0<=start<=STEPS: raise ValueError('Invalid checkpoint step')
            print(f'\n{variant} | FOLD {fold}/5 | starting at step {start}',flush=True)
            tick=time.monotonic()
            for step in range(start+1,STEPS+1):
                chosen=[tumor[j] for j in rng.choice(len(tumor),2,replace=False)]
                chosen += [healthy[int(rng.integers(len(healthy)))]]
                site=sites[int(rng.integers(len(sites)))]
                chosen += [pools[site][int(rng.integers(len(pools[site])))]]
                chosen=[chosen[j] for j in rng.permutation(4)]
                model.train();optimizer.zero_grad(set_to_none=True);total=0.
                for r in chosen:
                    loss=objective(model(bags[variant][r['id']].to(device)),torch.tensor([r['label']],device=device,dtype=torch.float32))/4
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();total+=float(loss.detach());exposures[r['id']]+=1
                optimizer.step()
                if step%100==0:
                    print(f'Step {step}/{STEPS} | batch loss {total:.4f} | last 100 steps {time.monotonic()-tick:.1f}s',flush=True)
                    tick=time.monotonic()
                    temp=state_file.with_suffix('.tmp')
                    torch.save(dict(identity=dict(experiment=experiment_hash,fold=fold,variant=variant),model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                                    rng_json=json.dumps(rng.bit_generator.state),exposures=dict(exposures)),temp)
                    temp.replace(state_file)
            trained_ids={r['id'] for r in train}
            if trained_ids & {r['id'] for r in test}:raise ValueError('Train/evaluation overlap')
            write_json(folder/'sampling.json',dict(exposures))
            model.eval();predictions=[]
            with torch.inference_mode():
                for r in test:
                    score=float(torch.sigmoid(model(bags[variant][r['id']].to(device))).item())
                    if not np.isfinite(score):raise ValueError('Nonfinite prediction')
                    predictions.append(dict(id=r['id'],source=r['source'],site=r['site'],label=r['label'],
                                            fold=fold,variant=variant,score=score,prediction=int(score>=.5)))
            write_json(folder/'predictions.json',predictions)
            all_predictions.extend(predictions)
            btc_metrics=metrics([r for r in predictions if r['source']=='BTC'])
            print('Held-out BTC:',json.dumps(btc_metrics),flush=True)
            write_json(folder/'checkpoint_protocol.json',dict(train_ids=sorted(trained_ids),
                evaluation_ids=[r['id'] for r in test],step=STEPS,threshold=.5,
                note='Final fixed step; no held-out checkpoint selection. Research only.'))
            del model,optimizer
            if device.type=='cuda':torch.cuda.empty_cache()
    report={'evaluation':'Exploratory reused BTC folds and IXI development CV; not untouched confirmation',
            'reserved_ixi_accessed':False,'mask_quality':'unverified; exploratory only','results':{},'paired_btc':{}}
    for variant in VARIANTS:
        subset=[r for r in all_predictions if r['variant']==variant]
        if len(subset)!=500 or len({r['id'] for r in subset})!=500:raise ValueError('Incomplete out-of-fold coverage')
        report['results'][variant]={}
        for group in ['BTC','IXI']+sites:
            selected=[r for r in subset if r['source']==group or r['site']==group]
            result=metrics(selected)
            report['results'][variant][group]=result
            print(variant,group,json.dumps(result),flush=True)
    baseline={r['id']:r for r in all_predictions if r['variant']=='voxel_cube' and r['source']=='BTC'}
    fixes=breaks=0
    for r in all_predictions:
        if r['variant']=='brain_extracted' and r['source']=='BTC':
            old=baseline[r['id']];before=old['prediction']==old['label'];after=r['prediction']==r['label']
            fixes+=int(not before and after);breaks+=int(before and not after)
    report['paired_btc']=dict(fixes=fixes,breaks=breaks)
    old=report['results']['voxel_cube']['BTC'];new=report['results']['brain_extracted']['BTC']
    report['paired_btc']['changes_percentage_points']={k:100*(new[k]-old[k]) for k in ('accuracy','balanced_accuracy','specificity','sensitivity')}
    lines=['EXPLORATORY PARTICIPANT-LEVEL RESULTS (percentages)']
    for variant,groups in report['results'].items():
        for group,m in groups.items():
            lines.append(variant+' / '+group+' | n='+str(m['n']))
            for k in ('accuracy','balanced_accuracy','sensitivity','specificity','precision','f1'):
                if k in m and m[k] is not None:lines.append(f'  {k}: {100*m[k]:.2f}%')
            lines.append('  confusion matrix: '+str(m['confusion_matrix']))
    lines.append('Paired BTC changes: '+json.dumps(report['paired_btc']))
    text='\n'.join(lines)
    print(text,flush=True)
    (OUT/'comparison.txt').write_text(text,encoding='utf-8')
    write_json(OUT/'comparison.json',report)
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_predictions[0]));w.writeheader();w.writerows(all_predictions)
    print('Primary comparison: BTC balanced accuracy and specificity, with tumor sensitivity. No automatic model adoption.',flush=True)
    print('Training comparison complete. No reserved test evaluation or deployment performed.',flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:
        run()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        (OUT/'error.txt').write_text(error,encoding='utf-8')
        raise
    finally:
        archive=BASE/'brain_extraction_comparison_report.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and 'cache' not in p.relative_to(OUT).parts and p.suffix in ('.json','.csv','.txt','.log','.png'):
                    z.write(p,str(p.relative_to(OUT)))
            for name in ('run_training.py','protocol.py','core.py','prepare.py','qc.py','README.txt','prior_review.json','extraction_source_lock.json'):
                z.write(BASE/name,name)
        print('UPLOAD THIS ZIP:',archive,flush=True)

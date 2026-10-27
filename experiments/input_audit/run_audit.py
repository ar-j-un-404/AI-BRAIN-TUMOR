import csv
import hashlib
import json
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
from audit_utils import sha,load_bag,statistics,montage
from protocol import metrics

BASE=Path(__file__).resolve().parent
VARIANTS=['voxel_cube','brain_extracted']

def dump(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(out):
    import torch
    from core import Classifier,seed_all
    torch.set_num_threads(2);seed_all(42)
    settings=json.loads((BASE/'settings.json').read_text())
    root=Path(settings['comparison_folder']);result=root/'results'
    config=json.loads((result/'experiment.json').read_text())
    if config['version']!='btc-ixi-extraction-paired-v1':raise ValueError('Wrong comparison experiment')
    if sha(BASE/'core.py')!=config['source_hashes']['core.py']:raise ValueError('Model code differs from training')
    experiment_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache_lock=json.loads((result/'training_cache_lock.json').read_text())
    records=config['records'];ids={r['id'] for r in records}
    btc=[r for r in records if r['source']=='BTC']
    if len(records)!=500 or len(ids)!=500 or len(btc)!=36:raise ValueError('Unexpected participant manifest')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,'| read-only audit of 36 BTC participants',flush=True)
    bags={v:{} for v in VARIANTS};stats=[]
    for r in btc:
        ident=r['id'];lock=cache_lock[ident]
        for v,suffix,key in [('voxel_cube','','baseline_sha256'),('brain_extracted','_brain','brain_sha256')]:
            bags[v][ident]=load_bag(result/'cache'/(ident+suffix+'.npy'),lock[key])
        row=dict(id=ident,label=r['label'],fold=r['fold'],**statistics(bags['voxel_cube'][ident],bags['brain_extracted'][ident]));stats.append(row)
    write_csv(out/'input_statistics.csv',stats)
    predictions=[];summaries=[];checkpoints=[];drifts=[];exposure_rows=[]
    for fold in range(1,6):
        expected_train={r['id'] for r in records if r['fold']!=fold}
        expected_test=ids-expected_train
        for variant in VARIANTS:
            print(f'Replaying {variant} / fold {fold}/5',flush=True)
            folder=result/variant/f'fold_{fold}'
            protocol=json.loads((folder/'checkpoint_protocol.json').read_text())
            if set(protocol['train_ids'])!=expected_train or set(protocol['evaluation_ids'])!=expected_test:
                raise ValueError('Fold membership mismatch')
            saved={r['id']:r for r in json.loads((folder/'predictions.json').read_text())}
            state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
            if state['identity']!=dict(experiment=experiment_hash,fold=fold,variant=variant) or state['step']!=1000:
                raise ValueError('Wrong checkpoint or incomplete training')
            model=Classifier().to(device);model.load_state_dict(state['model']);model.eval()
            checkpoints.append(dict(variant=variant,fold=fold,sha256=sha(folder/'resume.pt')))
            with torch.inference_mode():
                for r in btc:
                    role='held_out' if r['fold']==fold else 'training_subset'

                    input_variants=VARIANTS if role=='held_out' else [variant]
                    for input_variant in input_variants:
                        x=torch.from_numpy(bags[input_variant][r['id']])[None].to(device)
                        score=float(torch.sigmoid(model(x)).item())
                        if not np.isfinite(score):raise ValueError('Nonfinite score')
                        row=dict(id=r['id'],label=r['label'],fold=fold,model=variant,input=input_variant,role=role,score=score,prediction=int(score>=.5))
                        predictions.append(row)
                        if role=='held_out' and input_variant==variant:
                            drift=abs(score-saved[r['id']]['score']);drifts.append(drift)
                            if drift>1e-4 or int(score>=.5)!=saved[r['id']]['prediction']:
                                dump(out/'replay_mismatch.json',dict(**row,saved=saved[r['id']],absolute_drift=drift))
                                raise ValueError('Matched replay differs from saved result; inspect replay_mismatch.json before interpreting diagnostics')
                    exposure_rows.append(dict(id=r['id'],fold=fold,variant=variant,role=role,draws=state['exposures'].get(r['id'],0)))
            for role in ['training_subset','held_out']:
                group=[r for r in predictions if r['fold']==fold and r['model']==variant and r['input']==variant and r['role']==role]
                m=metrics(group)
                loss=float(np.mean([-r['label']*np.log(max(r['score'],1e-7))-(1-r['label'])*np.log(max(1-r['score'],1e-7)) for r in group]))
                summaries.append(dict(fold=fold,model=variant,role=role,bce_loss=loss,**m))
                print(f"  BTC {role}: accuracy {m['accuracy']*100:.2f}%, balanced {m['balanced_accuracy']*100:.2f}%, BCE {loss:.4f}",flush=True)
            del model,state
            if device.type=='cuda':torch.cuda.empty_cache()
    write_csv(out/'replayed_predictions.csv',predictions);write_csv(out/'training_exposures.csv',exposure_rows)
    dump(out/'fold_summaries.json',summaries);dump(out/'checkpoints.json',checkpoints)
    matched={(r['id'],r['model']):r for r in predictions if r['role']=='held_out' and r['input']==r['model']}
    changed=[]
    for r in btc:
        a=matched[(r['id'],'voxel_cube')];b=matched[(r['id'],'brain_extracted')]
        if a['prediction']!=b['prediction']:
            changed.append(dict(id=r['id'],label=r['label'],fold=r['fold'],original_score=a['score'],extracted_score=b['score'],
                                change='fix' if b['prediction']==r['label'] else 'break'))
    write_csv(out/'changed_cases.csv',changed)

    gallery=out/'model_inputs';gallery.mkdir()
    for r in btc:
        ident=r['id'];montage(bags['voxel_cube'][ident],bags['brain_extracted'][ident],gallery/(ident+'.png'),ident+' | cached model inputs; display axes exactly as tensor')
    summary=dict(completed=True,btc_participants=36,matched_held_out_replays=len(drifts),maximum_absolute_replay_drift=max(drifts),
                 changed_cases=changed,device=str(device),training_started=False,reserved_test_accessed=False,
                 interpretation='Training-subset metrics measure fitting, not generalization. Crossed model/input predictions are perturbation diagnostics, not deployable performance. No mask or tumor preservation approval.')
    dump(out/'summary.json',summary)
    lines=['READ-ONLY INPUT AUDIT COMPLETE',f'Matched held-out replays: {len(drifts)}; max score drift: {max(drifts):.8f}',
           'Fold 4 matched-input BTC results:']
    for r in summaries:
        if r['fold']==4:lines.append(f"{r['model']} / {r['role']}: accuracy {r['accuracy']*100:.2f}%, balanced {r['balanced_accuracy']*100:.2f}%, sensitivity {r['sensitivity']*100:.2f}%, specificity {r['specificity']*100:.2f}%, BCE {r['bce_loss']:.4f}")
    lines.append('No optimizer, retraining, checkpoint selection or threshold tuning was performed.')
    (out/'summary.txt').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':
    out=BASE/'reports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    try:run(out)
    except Exception:
        error=traceback.format_exc();(out/'error.txt').write_text(error);print(error,flush=True)
    finally:
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,str(p.relative_to(out)))
            for n in ['run_audit.py','audit_utils.py','core.py','protocol.py','README.txt']:z.write(BASE/n,'code/'+n)
        print('UPLOAD THIS ZIP:',archive,flush=True)

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(command,project,log):
    print('\nRUNNING:',subprocess.list2cmdline(command),flush=True)
    with log.open('a',encoding='utf-8') as f:
        p=subprocess.Popen(command,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,encoding='utf-8',errors='replace')
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line);f.flush()
        if p.wait():raise RuntimeError(f'Command failed; see {log}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=(Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'))
    args=parser.parse_args();project=args.project.resolve();suite=project/'research_suite'
    baseline=project/'runs/3d_full_all4_seed42'
    cfg=json.loads((baseline/'config.json').read_text())
    if cfg['task']!='segmentation3d' or cfg['channels']!=[0,1,2,3] or cfg['pilot']:
        raise ValueError('Expected the completed full four-modality model')
    if len(cfg['case_ids']['train'])!=338 or len(cfg['case_ids']['validation'])!=73:
        raise ValueError('Unexpected baseline case counts')

    for name in ('train3d.py','networks.py','engine.py','msd_data.py','metrics.py','common.py'):
        if not (suite/name).is_file() or sha(suite/name)!=cfg['code_sha256'].get(name):
            raise ValueError(f'{name} differs from the baseline implementation. Send this error for review.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this training run')
    sys.path.insert(0,str(suite))
    from msd_data import load_splits
    _,splits,split_hashes=load_splits(project)
    if split_hashes!=cfg['split_sha256']:raise ValueError('Split files changed')
    for partition in ('train','validation'):
        if [r['case_id'] for r in splits[partition]]!=cfg['case_ids'][partition]:
            raise ValueError('Case order differs from baseline')
    if not (baseline/'best.pt').is_file():raise FileNotFoundError('Baseline best.pt missing')
    output=project/'reports/modality_flair_vs_all4'
    output.mkdir(parents=True,exist_ok=True)
    model_output=project/'runs/3d_flair_matched_comparison'
    signature=dict(baseline_checkpoint_sha256=sha(baseline/'best.pt'),baseline_config=cfg,
        suite_sha256={p.name:sha(p) for p in suite.glob('*.py')},candidate_channels=[0],
        note='Development validation comparison. No internal-test evaluation. Different input channels change first-layer parameter count.')
    statepath=output/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(signature=signature,train_complete=False,evaluations={})
    if state['signature']!=signature:raise ValueError('Inputs/code changed since this comparison began')
    def save():statepath.write_text(json.dumps(state,indent=2))
    save()
    train=[sys.executable,str(suite/'train3d.py'),'--project',str(project),'--output',str(model_output),
           '--epochs',str(cfg['epochs']),'--steps',str(cfg['steps_per_epoch']),
           '--patch',*map(str,cfg['patch']),'--channels','0','--seed',str(cfg['seed']),
           '--lr',str(cfg['learning_rate']),'--validate-every',str(cfg['validate_every']),
           '--accumulate',str(cfg['accumulate'])]
    if not state['train_complete']:
        if model_output.exists():
            if not (model_output/'last.pt').exists():raise RuntimeError('Partial run has no last.pt; send training.log for review')
            train.append('--resume')
        run(train,project,output/'training.log')
        candidate=json.loads((model_output/'config.json').read_text())
        for key in ('case_ids','split_sha256','epochs','steps_per_epoch','patch','seed','learning_rate','validate_every','accumulate','base','dropout','protocol'):
            if candidate[key]!=cfg[key]:raise ValueError(f'Candidate setting mismatch: {key}')
        state['train_complete']=True;save()
    for name,checkpoint in (('all4',baseline/'best.pt'),('flair',model_output/'best.pt')):
        if name in state['evaluations']:
            if sha(checkpoint)!=state['evaluations'][name]['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            if not Path(state['evaluations'][name]['report']).is_file():raise FileNotFoundError('Completed report missing')
            continue

        dest=output/(name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        run([sys.executable,str(suite/'evaluate3d.py'),'--project',str(project),
             '--checkpoint',str(checkpoint),'--output',str(dest),'--mc-passes','0'],project,output/(name+'_evaluation.log'))
        report=dest/'metrics.json';data=json.loads(report.read_text())
        expected=set(cfg['case_ids']['validation']);actual=[r['case_id'] for r in data['cases']]
        if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Evaluation case mismatch')
        state['evaluations'][name]=dict(report=str(report),checkpoint_sha256=sha(checkpoint));save()
    comparison=output/('paired_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    run([sys.executable,str(suite/'compare_results.py'),'--baseline',state['evaluations']['all4']['report'],
         '--candidate',state['evaluations']['flair']['report'],'--output',str(comparison)],project,output/'comparison.log')
    summary={name:json.loads(Path(v['report']).read_text())['summary'] for name,v in state['evaluations'].items()}
    summary['note']='73 validation cases. Differences are FLAIR minus all4. Models selected using validation; not independent test results.'
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    archive=output/('modality_report_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [output/'summary.json',statepath,comparison]:z.write(p,p.name)
        for name,value in state['evaluations'].items():z.write(value['report'],name+'/metrics.json')
        z.write(model_output/'config.json','flair/config.json')
    print(json.dumps(summary,indent=2));print('UPLOAD THIS ZIP:',archive)


if __name__=='__main__':main()

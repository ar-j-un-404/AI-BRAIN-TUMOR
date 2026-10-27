import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from common import dump,read,sha,code_hashes

def main():
    p=argparse.ArgumentParser(description='One entry point for the runnable development workflow, with honest blocked-data status.')
    p.add_argument('--project',default='.');p.add_argument('--mode',choices=['check','pilot','full','ablations'],default='pilot')
    p.add_argument('--epochs',type=int);p.add_argument('--steps',type=int);p.add_argument('--patch',type=int,nargs=3,default=[64,96,96])
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--classification-manifest')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    project=Path(args.project).resolve();suite=Path(__file__).resolve().parent
    if not (project/'data/splits/split_manifest.json').is_file():raise FileNotFoundError('Wrong project folder: data/splits/split_manifest.json is missing. Use the INNER BrainTumor_AI_Stage2 folder.')
    run=project/f'runs/research_workflow_{args.mode}';run.mkdir(parents=True,exist_ok=True)
    statepath=run/'workflow.json';state=read(statepath) if statepath.exists() else {'jobs':{},'blocked':{}}
    implementation=code_hashes()
    def job(name,command):
        previous=state['jobs'].get(name)
        if previous and previous['status']=='complete':
            if not args.resume:raise ValueError(f'{name} already ran. Use --resume to retain completed jobs.')
            if previous['command']!=command or previous['code_sha256']!=implementation:raise ValueError('Workflow settings/code changed; preserve previous run and use separate output names')
            print('Already completed:',name,flush=True);return
        if previous and previous['status']!='complete':
            raise RuntimeError(f'Interrupted job {name}: see its log. Resume train3d.py directly with the exact printed arguments plus --resume; do not blindly replace a partial evaluation.')
        state['jobs'][name]={'status':'running','command':command,'code_sha256':implementation};dump(statepath,state)
        began=time.perf_counter();print('\nRUNNING:',subprocess.list2cmdline(command),flush=True)
        with (run/f'{name}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        state['jobs'][name].update(status='complete' if code==0 else 'failed',seconds=time.perf_counter()-began);dump(statepath,state)
        if code:raise RuntimeError(f'{name} failed. See {run/name}.log')
    job('synthetic_tests',[sys.executable,'-m','pytest',str(suite/'tests'),'-q'])
    if args.mode=='check':
        job('memory_check',[sys.executable,str(suite/'train3d.py'),'--smoke-only','--patch',*map(str,args.patch)])
        print('Software checks complete. No real MRI training performed.');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Fix the environment before launching real MRI training. Synthetic tests work on CPU.')
    epochs=args.epochs or (3 if args.mode=='pilot' else 20);steps=args.steps or (64 if args.mode=='pilot' else 512)
    variants={'all4':[0,1,2,3]}
    if args.mode=='ablations':variants={'flair':[0],'t1':[1],'t1ce':[2],'t2':[3],'t1_t2':[1,3],'all4':[0,1,2,3]}
    for seed in args.seeds:
        for variant,channels in variants.items():
            name=f'3d_{args.mode}_{variant}_seed{seed}';output=f'runs/{name}'
            command=[sys.executable,str(suite/'train3d.py'),'--project',str(project),'--output',output,'--epochs',str(epochs),
                '--steps',str(steps),'--patch',*map(str,args.patch),'--channels',*map(str,channels),'--seed',str(seed),
                '--validate-every',str(1 if args.mode=='pilot' else 5)]
            if args.mode=='pilot':command+=['--pilot']
            job(name,command)
            ev=f'reports/{name}'
            job(name+'_evaluate',[sys.executable,str(suite/'evaluate3d.py'),'--project',str(project),
                '--checkpoint',str(project/output/'best.pt'),'--output',ev,'--mc-passes','2' if args.mode=='pilot' else '4'])
            job(name+'_plots',[sys.executable,str(suite/'plot_reports.py'),'--report',str(project/ev/'metrics.json'),'--output',str(project/ev/'figures')])
    if args.classification_manifest:
        job('classification',[sys.executable,str(suite/'classify.py'),'train','--manifest',str(Path(args.classification_manifest).resolve()),
            '--output',str(project/'runs/classification_001'),'--epochs','20'])
    else:state['blocked']['classification']='Needs verified healthy/tumor participants with both classes within each acquisition source; see templates and data documentation.'
    state['blocked']['external_evaluation']='Requires a reviewed independent cohort, compatible labels and explicit frozen model selection. Not automatically evaluated during development.'
    state['blocked']['nnunet']='Separate optional reference environment and benchmark; see docs/NNUNET.md.'
    dump(statepath,state)
    print('\nDEVELOPMENT WORKFLOW COMPLETE. Test images were not opened.\nRemaining data/reference work:')
    print(json.dumps(state['blocked'],indent=2));print('App: python -m streamlit run research_suite/app.py --server.port 8505 --server.address localhost')
if __name__=='__main__':main()

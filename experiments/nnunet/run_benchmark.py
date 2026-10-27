import argparse,json,os,shutil,subprocess,sys,traceback,zipfile,hashlib
from pathlib import Path
from datetime import datetime

PROJECT=(Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def run(cmd,env,log,cwd):
    print('\nRUNNING:',subprocess.list2cmdline(list(map(str,cmd))),flush=True)
    with log.open('a',encoding='utf-8') as f:
        process=subprocess.Popen(list(map(str,cmd)),env=env,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:print(line,end='',flush=True);f.write(line);f.flush()
        if process.wait():raise RuntimeError('Command failed; see benchmark.log')

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=PROJECT);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.plan_only:
        print('Verify saved plans/splits -> preprocess 3d_fullres with 2 workers -> train fold 0 for 250 epochs -> final-checkpoint validation -> WT/TC/ET and paired comparison. No internal-test inference.');return 0
    project=a.project.resolve();base=Path(__file__).resolve().parent
    workspace=project/'nnunet_reference_v1';pre=workspace/'preprocessed/Dataset501_MSDReviewed'
    raw=workspace/'raw/Dataset501_MSDReviewed'
    model=workspace/'results/Dataset501_MSDReviewed/nnUNetTrainer_250epochs__nnUNetPlans_2GB__3d_fullres/fold_0'
    output=project/'reports/nnunet_250epoch_2gb';output.mkdir(parents=True,exist_ok=True)
    log=output/'benchmark.log';statefile=output/'state.json';ok=False
    env=os.environ.copy();env.update(nnUNet_raw=str(workspace/'raw'),nnUNet_preprocessed=str(workspace/'preprocessed'),nnUNet_results=str(workspace/'results'),nnUNet_compile='false',nnUNet_n_proc_DA='2',OMP_NUM_THREADS='2',PYTHONHASHSEED='42')
    python=project/'.venv_nnunet/Scripts/python.exe';original=project/'.venv/Scripts/python.exe'
    state={}
    def save():
        tmp=statefile.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(statefile)
    try:
        for f in (python,original,project/'research_suite/metrics.py',project/'reports/3d_full_all4_seed42/metrics.json'):
            if not f.is_file():raise FileNotFoundError(f)
        for f in ('nnUNetPlans_2GB.json','splits_final.json','dataset.json'):
            if json.loads((pre/f).read_text())!=json.loads((base/'verified_inputs'/f).read_text()):raise ValueError(f'Previously verified {f} changed')
        if json.loads((raw/'dataset.json').read_text())!=json.loads((pre/'dataset.json').read_text()):raise ValueError('Raw dataset metadata changed')

        code="import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from msd_data import load_splits; _,s,h=load_splits(Path(sys.argv[2])); expected=[{'train':[r['case_id'] for r in s['train']],'val':[r['case_id'] for r in s['validation']]}]; assert json.load(open(sys.argv[3]))==expected; assert not set(expected[0]['train']) & set(expected[0]['val']); print('Original split IDs verified')"
        run([original,'-c',code,project/'research_suite',project,pre/'splits_final.json'],env,log,base)
        version=subprocess.check_output([str(python),'-c',"import importlib.metadata as m; print(m.version('nnunetv2'))"],text=True,env=env).strip()
        if version!='2.8.1':raise ValueError('Expected installed nnunetv2 2.8.1')
        signature={'plans':sha(pre/'nnUNetPlans_2GB.json'),'splits':sha(pre/'splits_final.json'),'nnunet':version,'epochs':250,'checkpoint':'final',
            'scripts':{f.name:sha(f) for f in base.glob('*.py')},'region_metrics':sha(project/'research_suite/metrics.py'),'baseline_report':sha(project/'reports/3d_full_all4_seed42/metrics.json')}
        state=json.loads(statefile.read_text()) if statefile.exists() else {'signature':signature,'preprocessed':False,'trained':False,'scored':False}
        if state['signature']!=signature:raise ValueError('Benchmark inputs changed. Preserve this run for review.')
        save()
        (output/'environment.txt').write_text(subprocess.check_output([str(python),'-m','pip','freeze'],text=True,env=env))
        if not state['preprocessed']:
            free=shutil.disk_usage(workspace).free/2**30
            if free<40:raise RuntimeError(f'Only {free:.1f} GiB free. Preprocessing needs space; send report before continuing.')
            run([python.parent/'nnUNetv2_preprocess.exe','-d','501','-plans_name','nnUNetPlans_2GB','-c','3d_fullres','-np','2'],env,log,base)
            if json.loads((pre/'splits_final.json').read_text())!=json.loads((base/'verified_inputs/splits_final.json').read_text()):raise ValueError('Preprocessing changed split')
            state['preprocessed']=True;save()
        if not state['trained']:
            args=[python,base/'train_entry.py','501','3d_fullres','0','-tr','nnUNetTrainer_250epochs','-p','nnUNetPlans_2GB','-device','cuda']
            if (model/'checkpoint_final.pth').exists():args.append('--val')
            elif (model/'checkpoint_latest.pth').exists():args.append('--c')
            elif (model/'checkpoint_best.pth').exists():raise RuntimeError('Only best checkpoint exists; review resume before continuing.')
            run(args,env,log,base)
            if not (model/'checkpoint_final.pth').is_file():raise FileNotFoundError('Final checkpoint missing after training')
            state['trained']=True;state['final_checkpoint_sha256']=sha(model/'checkpoint_final.pth');save()
        if sha(model/'checkpoint_final.pth')!=state['final_checkpoint_sha256']:raise ValueError('Final checkpoint changed')
        if not state['scored']:
            run([original,base/'evaluate_regions.py','--project',project,'--predictions',model/'validation','--output',output/'metrics.json'],env,log,base)
            baseline=project/'reports/3d_full_all4_seed42/metrics.json'
            run([original,base/'compare_regions.py','--project',project,'--baseline',baseline,'--candidate',output/'metrics.json','--output',output/'paired_vs_small3d.json'],env,log,base)
            state['scored']=True;save()
        ok=True
    except Exception:
        error=traceback.format_exc();print(error);(output/'latest_error.txt').write_text(error)
    finally:
        (output/'status.json').write_text(json.dumps({'complete':ok,'development_only':True,'internal_test_evaluated':False},indent=2))
        archive=output/('nnunet_report_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in output.iterdir():
                if f.suffix in ('.json','.txt','.log'):z.write(f,f.name)
            for f in ('nnUNetPlans_2GB.json','splits_final.json','dataset.json'):
                if (pre/f).is_file():z.write(pre/f,'configuration/'+f)
            for f in model.glob('*.png'):z.write(f,'training/'+f.name)
        print('UPLOAD THIS ZIP:',archive,flush=True)
    return 0 if ok else 1
if __name__=='__main__':sys.exit(main())

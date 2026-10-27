import json
import subprocess
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    settings_file=BASE/'settings.json'
    if settings_file.exists():settings=json.loads(settings_file.read_text())
    else:
        project=Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'
        if not (project/'.venv/Scripts/python.exe').is_file():
            project=Path(input('Paste main BrainTumor project folder: ').strip().strip('"'))
        candidates=[]
        for root in (Path.home()/'Downloads').glob('BrainTumor_Brain_Extraction_QC*'):
            candidates.extend(root.glob('assets/manifest.json'))
            candidates.extend(root.glob('*/assets/manifest.json'))
        if candidates:
            selected=max(candidates,key=lambda p:p.stat().st_mtime_ns).parent.parent
            print('Using completed extraction folder:',selected,flush=True)
        else:selected=Path(input('Paste completed BrainTumor_Brain_Extraction_QC folder (contains worker.py and assets): ').strip().strip('"'))
        settings=dict(project=str(project),extraction_folder=str(selected),worker_python=str(project/'.venv_nnunet/Scripts/python.exe'))
        settings_file.write_text(json.dumps(settings,indent=2))
    python=Path(settings['project'])/'.venv/Scripts/python.exe'
    worker=Path(settings['worker_python'])
    if not python.is_file() or not worker.is_file():raise FileNotFoundError('Project Python environments missing; correct settings.json.')
    if not (Path(settings['extraction_folder'])/'assets/manifest.json').is_file():raise FileNotFoundError('Completed extraction folder missing; correct settings.json.')
    print('Close other GPU training or model inference. Existing app files will not be changed.',flush=True)
    print('500 development participants only; 117 reserved IXI scans remain unused.',flush=True)
    print('Mask quality remains unverified. This is an exploratory experiment.',flush=True)
    check=subprocess.run([str(worker),'-c','import torch, SimpleITK; from importlib.metadata import version; assert version("nnunetv2")=="2.8.1"; print("Extraction environment ready")'])
    if check.returncode:raise RuntimeError('Extraction environment check failed')
    check=subprocess.run([str(python),str(BASE/'runtime_check.py')],cwd=BASE)
    if check.returncode:raise RuntimeError('Training runtime check failed; no new training started')
    return subprocess.call([str(python),'-u',str(BASE/'run_training.py')],cwd=BASE)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('ERROR:',exc,flush=True);sys.exit(1)

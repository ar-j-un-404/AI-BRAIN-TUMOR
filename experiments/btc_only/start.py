import json
import subprocess
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
try:
    settings=BASE/'settings.json'
    if not settings.exists():
        project=Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'
        if not (project/'.venv/Scripts/python.exe').exists():project=Path(input('Main project folder: ').strip().strip('"'))
        candidates=[]
        for folder in (Path.home()/'Downloads').glob('BrainTumor_Extraction_Comparison*'):
            candidates.extend(folder.glob('results/comparison.json'));candidates.extend(folder.glob('*/results/comparison.json'))
        if candidates:comparison=max(candidates,key=lambda p:p.stat().st_mtime_ns).parent.parent
        else:comparison=Path(input('Completed BrainTumor_Extraction_Comparison folder: ').strip().strip('"'))
        settings.write_text(json.dumps(dict(project=str(project),comparison_folder=str(comparison)),indent=2))
    config=json.loads(settings.read_text());python=Path(config['project'])/'.venv/Scripts/python.exe'
    if not python.exists():raise FileNotFoundError(python)
    print('Reading comparison:',config['comparison_folder'],flush=True)
    print('Five new BTC-only training runs. Close other GPU training/inference.',flush=True)
    sys.exit(subprocess.call([str(python),'-u',str(BASE/'run_training.py')],cwd=BASE))
except Exception as e:print('ERROR:',e,flush=True);sys.exit(1)

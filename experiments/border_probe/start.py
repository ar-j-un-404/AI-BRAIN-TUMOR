import json
import subprocess
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
project=Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'
try:
    python=project/'.venv/Scripts/python.exe'
    if not python.is_file():
        project=Path(input('Paste your main BrainTumor project folder: ').strip().strip('"'))
        python=project/'.venv/Scripts/python.exe'
    if not python.is_file():raise FileNotFoundError(python)
    candidates=[]
    for top in (Path.home()/'Downloads').glob('BrainTumor_Physical_Space_v2*'):
        if not top.is_dir():continue
        for p in [top/'results',*(d/'results' for d in top.iterdir() if d.is_dir())]:
            if (p/'experiment.json').is_file() and (p/'cache').is_dir():candidates.append(p)
    candidates=sorted(set(candidates))
    if len(candidates)==1:root=candidates[0]
    else:
        print('Choose the completed physical-space V2 results folder containing cache and experiment.json.')
        for p in candidates:print(p)
        root=Path(input('Paste results folder: ').strip().strip('"'))
    check=subprocess.run([str(python),'-c','import numpy,sklearn,PIL; print("CPU dependencies ready")'])
    if check.returncode:
        print('Install missing packages using:')
        print('"'+str(python)+'" -m pip install scikit-learn pillow')
        raise RuntimeError('Dependencies missing')
    print('Using cached MRI inputs:',root,flush=True)
    print('CPU diagnostic. No long CNN training, no GPU needed.',flush=True)
    sys.exit(subprocess.run([str(python),'-u',str(base/'run_probe.py'),'--results',str(root)],cwd=base).returncode)
except Exception as exc:
    print('ERROR:',exc,flush=True);sys.exit(1)

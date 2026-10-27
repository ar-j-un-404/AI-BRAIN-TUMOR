import subprocess
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
project=Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'
try:
    python=project/'.venv/Scripts/python.exe'
    if not python.is_file():
        project=Path(input('Paste your main project folder: ').strip().strip('"'))
        python=project/'.venv/Scripts/python.exe'
    if not python.is_file():raise FileNotFoundError(python)
    print('Close other GPU training and app inference before continuing.',flush=True)
    print('Ten runs: 2 preprocessing protocols x 5 identical participant folds.',flush=True)
    print('Existing app and checkpoints are not modified. Restart this launcher to resume.',flush=True)
    check=subprocess.run([str(python),'-c','import torch,numpy,nibabel,scipy; print("CUDA available:",torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA required for this launcher"'],cwd=base)
    if check.returncode:raise RuntimeError('Environment check failed. Use your working project environment.')
    sys.exit(subprocess.run([str(python),'-u',str(base/'run_training.py')],cwd=base).returncode)
except Exception as e:
    print('ERROR:',e,flush=True)
    sys.exit(1)

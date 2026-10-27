import json
import subprocess
import sys
from pathlib import Path

BASE=Path(__file__).resolve().parent

def main():
    project=Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'
    if not (project/'.venv_nnunet/Scripts/python.exe').is_file():
        project=Path(input('Paste your main project folder containing .venv and .venv_nnunet: ').strip().strip('"'))
    mainpython=project/'.venv/Scripts/python.exe';nnpython=project/'.venv_nnunet/Scripts/python.exe'
    for p in [mainpython,nnpython]:
        if not p.is_file():raise FileNotFoundError(p)
    subprocess.run([str(mainpython),'-c','import nibabel,numpy,scipy,PIL'],check=True)
    subprocess.run([str(nnpython),'-c',"import torch,SimpleITK,nnunetv2; from importlib.metadata import version; assert version('nnunetv2')=='2.8.1', 'Expected reviewed nnunetv2 2.8.1'; print('CUDA:',torch.cuda.is_available())"],check=True)
    print('First run downloads official HD-BET weights. Close other GPU jobs.',flush=True)
    subprocess.run([sys.executable,str(BASE/'setup_assets.py')],check=True,cwd=BASE)
    result=subprocess.run([str(mainpython),'-u',str(BASE/'run_qc.py'),'--worker-python',str(nnpython)],cwd=BASE)
    return result.returncode

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('SETUP ERROR:',e,flush=True);sys.exit(1)

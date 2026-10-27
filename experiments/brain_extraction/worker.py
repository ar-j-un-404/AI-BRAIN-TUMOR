import argparse
import json
import os
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    base=Path(__file__).resolve().parent
    sys.path.insert(0,str(base/'vendor'))
    os.environ['nnUNet_compile']='false';os.environ['OMP_NUM_THREADS']='2'
    import torch
    import HD_BET.paths
    assets=json.loads((base/'assets/manifest.json').read_text())
    HD_BET.paths.folder_with_parameter_files=assets['model_folder']
    from HD_BET.hd_bet_prediction import get_hdbet_predictor
    torch.set_num_threads(2)
    device=a.device if torch.cuda.is_available() else 'cpu'
    try:
        predictor=get_hdbet_predictor(use_tta=False,device=torch.device(device),verbose=False)

        predictor.perform_everything_on_device=False
        torch.set_num_threads(2)
        predictor.predict_from_files([[a.input]],[a.output],save_probabilities=False,overwrite=True,
            num_processes_preprocessing=1,num_processes_segmentation_export=1)
        Path(a.output+'.device.json').write_text(json.dumps({'device':device,'tta':False,'host_accumulation':True}))
    except RuntimeError as exc:
        if device=='cuda' and 'out of memory' in str(exc).lower():
            print('GPU_OUT_OF_MEMORY: retry on CPU',flush=True);sys.exit(42)
        raise

if __name__=='__main__':main()

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from common import dump
from msd_data import load_splits,prepare_case

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Export the existing train/validation cases; never export internal-test scans.'); p.add_argument('--project',default='.')
    p.add_argument('--destination',default='nnunet_workspace'); args=p.parse_args()
    project=Path(args.project).resolve(); root,splits,hashes=load_splits(project)
    dest=(project/args.destination).resolve(); name='Dataset501_MSDReviewed'
    raw=dest/'raw'/name
    if raw.exists(): raise ValueError('Export exists. Do not overwrite a prior benchmark.')
    (raw/'imagesTr').mkdir(parents=True); (raw/'labelsTr').mkdir()
    training=splits['train']+splits['validation']
    for i,row in enumerate(training,1):
        print(f'Export {i}/{len(training)}: {row["case_id"]}',flush=True)
        prepare_case(root,row,project/'data/stage3_cache_v1')
        image=nib.load(root/row['image']); mask=nib.load(root/row['mask'])
        for channel in range(4): nib.save(image.slicer[:,:,:,channel],raw/'imagesTr'/f'{row["case_id"]}_{channel:04d}.nii.gz')
        nib.save(mask,raw/'labelsTr'/f'{row["case_id"]}.nii.gz')
    dump(raw/'dataset.json',{'channel_names':{'0':'FLAIR','1':'T1','2':'T1ce','3':'T2'},
        'labels':{'background':0,'edema':1,'non-enhancing tumor':2,'enhancing tumour':3},
        'numTraining':len(training),'file_ending':'.nii.gz'})
    pre=dest/'preprocessed'/name; pre.mkdir(parents=True)
    dump(pre/'splits_final.json',[{'train':[r['case_id'] for r in splits['train']], 'val':[r['case_id'] for r in splits['validation']]}])
    dump(dest/'export_provenance.json',{'split_sha256':hashes,'internal_test_exported':False,
        'note':'One custom fold (fold 0). nnU-Net development planning uses all exported train+validation cases; not strict training-only preprocessing.'})
    (dest/'ENV_WINDOWS.bat').write_text(f'@echo off\nset "nnUNet_raw={dest / "raw"}"\nset "nnUNet_preprocessed={dest / "preprocessed"}"\nset "nnUNet_results={dest / "results"}"\n',encoding='utf-8')
    print('Export complete. See docs/NNUNET.md before installing/planning.')

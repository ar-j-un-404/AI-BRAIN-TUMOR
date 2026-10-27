import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
import nibabel as nib
import numpy as np
from protocol import load_records
from qc import select_cases,prepare_input,check_mask,montage

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')

def run(python,out):
    records=select_cases(load_records(BASE));assert len(records)==45
    assets=json.loads((BASE/'assets/manifest.json').read_text())
    protocol=dict(version='hdbet-qc-v1',tta=False,orientation='LAS axis reorder only; no registration',
        precision='official HD-BET predictor defaults',planned_device='cuda with recorded CPU fallback',
        source_hashes={n:sha(BASE/n) for n in ['run_qc.py','worker.py','qc.py','protocol.py']},assets=assets)
    dump(out/'protocol.json',protocol);dump(out/'selected_cases.json',records)
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    work=BASE/'work';work.mkdir(exist_ok=True);results=[];device='cuda'
    for i,r in enumerate(records,1):
        print(f"Brain extraction {i}/{len(records)}: {r['id']}",flush=True)
        case=work/r['id'];case.mkdir(exist_ok=True)
        try:
            source=Path(r['path']);input_hash=sha(source)
            las=case/'input_las.nii.gz';image=prepare_input(r,las)
            rawmask=case/'predicted_mask.nii.gz';done=case/'done.json'
            key=dict(protocol=signature,input_sha256=input_hash)
            cached=json.loads(done.read_text()) if done.is_file() else {}
            reusable=cached.get('key')==key and rawmask.is_file() and cached.get('mask_sha256')==sha(rawmask)
            log=out/(r['id']+'.log')
            if not reusable:
                command=[str(python),'-u',str(BASE/'worker.py'),'--input',str(las),'--output',str(rawmask),'--device',device]
                env=os.environ.copy();env['nnUNet_compile']='false'
                with log.open('w',encoding='utf-8') as f:
                    code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
                    if code==42:
                        print('GPU memory exhausted; continuing on CPU.',flush=True);device='cpu';command[-1]='cpu'
                        code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
                if code:raise RuntimeError('HD-BET worker failed; see '+log.name)
            mask,stats=check_mask(image,nib.load(rawmask))
            a=image.get_fdata(dtype=np.float32)
            brain=case/'brain_ras.nii.gz';maskpath=case/'mask_ras.nii.gz'
            nib.save(nib.Nifti1Image(np.where(mask,a,0).astype('float32'),image.affine,image.header),brain)
            header=image.header.copy();header.set_data_dtype('uint8')
            nib.save(nib.Nifti1Image(mask.astype('uint8'),image.affine,header),maskpath)
            montage(image,mask,out/(r['id']+'.png'),r['id']+' / '+r['source'])
            device_record=json.loads(Path(str(rawmask)+'.device.json').read_text())
            if device_record['device']=='cpu':device='cpu'
            row=dict(id=r['id'],source=r['source'],label=r['label'],status='visual_review_pending',
                     input_sha256=input_hash,mask_sha256=sha(maskpath),brain_path=str(brain),mask_path=str(maskpath),
                     device=device_record['device'],reused=reusable,**stats)
            dump(done,dict(key=key,mask_sha256=sha(rawmask)))

            las.unlink(missing_ok=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            (out/(r['id']+'_error.txt')).write_text(error)
            row=dict(id=r['id'],source=r['source'],status='failed',error=error)
        results.append(row);dump(out/'cases.json',results)
        if row['status']=='failed' and 'HD-BET worker failed' in row.get('error',''):
            print('Stopping after a worker failure; inspect the report before retrying all cases.',flush=True)
            break
    review=out/'review.csv'
    with review.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','review_status','brain_boundary_ok','tumor_preserved_if_present','notes']);w.writeheader()
        for r in records:w.writerow(dict(id=r['id'],review_status='pending'))
    good=sum(r['status']=='visual_review_pending' for r in results)
    dump(out/'summary.json',dict(selected=45,processed=good,failed=len(results)-good,not_attempted=45-len(results),
        flagged=sum(bool(r.get('automatic_flags')) for r in results),visual_review_pending=good,
        training_started=False,reserved_test_accessed=False,
        note='Mask review required. Automated checks cannot prove anatomy or tumor preservation. No classifier retrained.'))
    print(f'QC COMPLETE: {good}/45 masks ready for review. No classifier training started.',flush=True)
    return 0 if good==45 else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker-python',type=Path,required=True);args=parser.parse_args()
    out=BASE/'reports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    code=1
    try:code=run(args.worker_python,out)
    except Exception:(out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():
                if p.is_file():z.write(p,p.name)
            for name in ['run_qc.py','qc.py','worker.py','setup_assets.py']:z.write(BASE/name,'code/'+name)
        print('UPLOAD THIS ZIP:',archive,flush=True)
    sys.exit(code)

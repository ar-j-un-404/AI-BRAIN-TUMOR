import hashlib
import json
import os
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
from qc import prepare_input, check_mask, montage

PROTOCOL = 'HD-BET2.0.1-noTTA-LAS; normalize inside mask; same acquired voxel cube96 and 24 axial slices; no brain bounding-box crop'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(p)

def to_bag(a, mask):
    import torch
    from torch.nn import functional as F
    values=a[mask]
    if len(values)<1000 or values.std()<1e-8:raise ValueError('Insufficient masked intensity variation')
    out=np.zeros_like(a,dtype=np.float32)
    out[mask]=np.clip((values-values.mean())/values.std(),-5,5)
    side=max(out.shape)
    out=np.pad(out,[((side-n)//2,side-n-(side-n)//2) for n in out.shape])
    v=F.interpolate(torch.from_numpy(out)[None,None],size=(96,96,96),mode='trilinear',align_corners=False)[0,0]
    return v[:,:,torch.linspace(14,81,24).round().long()].permute(2,0,1).unsqueeze(1).contiguous().numpy()

def verify_assets(base, old):
    locks=json.loads((base/'extraction_source_lock.json').read_text())
    for n,h in locks.items():
        if sha(old/n)!=h:raise ValueError('Extraction software differs from verified run: '+n)
    assets=json.loads((old/'assets/manifest.json').read_text())
    root=old/'assets/model'
    if not Path(assets['model_folder']).resolve().is_relative_to(root.resolve()):raise ValueError('Weights path is outside verified model directory')
    for n,h in assets['files'].items():
        if sha(root/n)!=h:raise ValueError('HD-BET model file changed: '+n)
    if assets.get('source_sha256')!='46e3ad50b0c8d45b2b91cf2d2b821b3d26fd885ffa31628ad7687b5651a4838b':
        raise ValueError('Unexpected HD-BET source')
    if assets.get('model_archive_md5')!='f97a36f89f8eb3071ff05c6f9ac66b56':raise ValueError('Unexpected weights')

    import tarfile
    archive=old/'assets/hd_bet-2.0.1.tar.gz'
    if sha(archive)!=assets['source_sha256']:raise ValueError('Source archive changed')
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            parts=Path(member.name).parts[1:]
            if member.isfile() and parts and parts[0]=='HD_BET':
                if (old/'vendor').joinpath(*parts).read_bytes()!=t.extractfile(member).read():
                    raise ValueError('HD-BET vendor file changed')
    protocol=dict(version='hdbet-qc-v1',tta=False,orientation='LAS axis reorder only; no registration',
                  precision='official HD-BET predictor defaults',planned_device='cuda with recorded CPU fallback',
                  source_hashes=locks,assets=assets)
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return assets,signature

def prepare_all(base,out,records):
    settings=json.loads((base/'settings.json').read_text())
    old=Path(settings['extraction_folder']); worker_python=Path(settings['worker_python'])
    assets,signature=verify_assets(base,old)
    provenance=dict(assets=assets,extraction_signature=signature,mask_quality='unverified',protocol=PROTOCOL)
    frozen=out/'extraction_protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=provenance:raise ValueError('Extraction protocol changed since this run')
    dump(frozen,provenance)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    work=base/'work';work.mkdir(exist_ok=True)
    logs=out/'extraction_logs';logs.mkdir(exist_ok=True)
    previews=out/'previews';previews.mkdir(exist_ok=True)
    from qc import select_cases
    preview_ids={r['id'] for r in select_cases(records)}
    reviews={r['id']:r for r in json.loads((base/'prior_review.json').read_text())}
    rows=[];seen={};device='cuda'
    for i,r in enumerate(records,1):
        print(f"Preparation {i}/{len(records)}: {r['id']}",flush=True)
        src=Path(r['path']); input_hash=sha(src)
        case=work/r['id'];case.mkdir(exist_ok=True)
        las=case/'input_las.nii.gz'

        image=prepare_input(r,las)
        a=image.get_fdata(dtype=np.float32)
        fingerprint=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        if fingerprint in seen:raise ValueError('Duplicate image content across participants')
        seen[fingerprint]=r['id']
        oldcase=old/'work'/r['id']; oldmask=oldcase/'predicted_mask.nii.gz'
        olddone=oldcase/'done.json'
        rawmask=case/'predicted_mask.nii.gz';done=case/'done.json'
        reuse=False; origin='new'
        key=dict(protocol=signature,input_sha256=input_hash)
        for folder, candidate, marker, label in [(oldcase,oldmask,olddone,'previous_extraction'),(case,rawmask,done,'resumed_extraction')]:
            if marker.exists() and candidate.exists():
                d=json.loads(marker.read_text())
                if d.get('key')==key and sha(candidate)==d.get('mask_sha256'):
                    rawmask=candidate;reuse=True;origin=label;break
        if not reuse:
            command=[str(worker_python),'-u',str(old/'worker.py'),'--input',str(las),'--output',str(rawmask),'--device',device]
            env=os.environ.copy();env['nnUNet_compile']='false'
            with (logs/(r['id']+'.log')).open('w',encoding='utf-8') as f:
                code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
                if code==42:
                    device='cpu';command[-1]='cpu'
                    print('GPU memory exhausted. Continuing extraction on CPU.',flush=True)
                    code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
            if code:raise RuntimeError('Extraction failed: '+r['id']+'. No cases silently excluded; see log.')
            dump(done,dict(key=key,mask_sha256=sha(rawmask)))
        mask,stats=check_mask(image,nib.load(rawmask))
        mask_hash=sha(rawmask)
        marker=cache/(r['id']+'.json')
        file=cache/(r['id']+'.npy');brain_file=cache/(r['id']+'_brain.npy')
        cache_key=dict(input_sha256=input_hash,mask_sha256=mask_hash,protocol=PROTOCOL)
        cached=json.loads(marker.read_text()) if marker.exists() else {}
        if cached.get('key')==cache_key and file.exists() and brain_file.exists():
            if sha(file)!=cached['baseline_sha256'] or sha(brain_file)!=cached['brain_sha256']:
                raise ValueError('Cached training data changed: '+r['id'])
        else:
            np.save(file,to_bag(a,a!=0),allow_pickle=False)
            np.save(brain_file,to_bag(a,mask),allow_pickle=False)
            dump(marker,dict(key=cache_key,baseline_sha256=sha(file),brain_sha256=sha(brain_file)))
        if r['id'] in preview_ids or stats['automatic_flags']:
            montage(image,mask,previews/(r['id']+'.png'),r['id']+' / UNVERIFIED MASK')
        decision=reviews.get(r['id'],{}).get('status','pending')
        row=dict(id=r['id'],source=r['source'],site=r['site'],label=r['label'],fold=r['fold'],
                 extraction_origin=origin,review_status=decision,input_sha256=input_hash,
                 raw_mask_sha256=mask_hash,mask_path=str(rawmask),**stats)
        rows.append(row);dump(out/'extraction_cases.json',rows)
        las.unlink(missing_ok=True)
    dump(out/'extraction_summary.json',dict(processed=len(rows),flags=sum(bool(x['automatic_flags']) for x in rows),
         reused=sum(x['extraction_origin']!='new' for x in rows),mask_quality='unverified',
         note='Exploratory comparison; flagged cases retained, no automatic anatomical approvals.'))
    return cache

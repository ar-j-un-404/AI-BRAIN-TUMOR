import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import dump, read, sha, seed_all, device, save_checkpoint, code_hashes, load_checkpoint
from msd_data import load_splits, prepare_case
from networks import UNet3D, segmentation_loss
from volumes import Patches, sliding
from metrics import region_metrics, summarize

def smoke(patch=(64,96,96), channels=4):
    dev = device(); seed_all(42); model = UNet3D(channels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1,channels,*patch,device=dev); y = torch.randint(0,4,(1,*patch),device=dev)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type=='cuda')
    with torch.autocast(device_type=dev.type, enabled=dev.type=='cuda', dtype=torch.float16):
        loss = segmentation_loss(model(x), y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval()
    with torch.inference_mode(): z = model(x)
    if dev.type=='cuda': torch.cuda.synchronize()
    return {'device':str(dev), 'gpu':torch.cuda.get_device_name(0) if dev.type=='cuda' else None,
        'parameters':sum(p.numel() for p in model.parameters()), 'patch':list(patch), 'shape':list(z.shape),
        'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if dev.type=='cuda' else None,
        'synthetic_forward_backward_optimizer_and_fp32_inference':'passed', 'real_mri_training':False}

def train(args):
    project = Path(args.project).resolve(); out = project/args.output
    patch = tuple(args.patch); channels = args.channels
    if len(set(channels))!=len(channels) or not set(channels)<=set(range(4)): raise ValueError('Invalid modality channels')
    if min(patch)<16 or any(p%8 for p in patch): raise ValueError('Patch dimensions must be >=16 and divisible by 8')
    if min(args.epochs,args.steps,args.validate_every,args.accumulate)<1: raise ValueError('Training counts must be positive')
    print('GPU memory smoke check...', flush=True); print(smoke(patch,len(channels)), flush=True)
    root, splits, split_hashes = load_splits(project)
    selected = {s: splits[s][:args.pilot_train if s=='train' else args.pilot_validation] if args.pilot else splits[s]
                for s in ['train','validation']}
    config = {'task':'segmentation3d', 'protocol':'msd-normalized-native-1mm-3d-v1',
        'channels':channels,'patch':list(patch),'base':8,'dropout':.1,'seed':args.seed,
        'epochs':args.epochs,'steps_per_epoch':args.steps,'validate_every':args.validate_every,
        'learning_rate':args.lr,'accumulate':args.accumulate,'pilot':args.pilot,
        'case_ids':{s:[r['case_id'] for r in selected[s]] for s in selected},
        'split_sha256':split_hashes,'code_sha256':code_hashes(),'torch_version':str(torch.__version__)}
    if out.exists():
        if not args.resume: raise ValueError(f'Output already exists: {out}. Use --resume or a new output name.')
        if read(out/'config.json')!=config: raise ValueError('Resume configuration/code/split mismatch')
    else: out.mkdir(parents=True); dump(out/'config.json',config)
    folders = {}; fingerprints = {}
    for partition, records in selected.items():
        folders[partition] = []
        for i,row in enumerate(records,1):
            print(f'Verifying {partition} cache {i}/{len(records)}: {row["case_id"]}', flush=True)
            info = prepare_case(root,row,project/'data/stage3_cache_v1')
            if not np.allclose(info['spacing'],[1,1,1],atol=1e-3): raise ValueError('This protocol requires reviewed 1-mm isotropic scans')
            fingerprints[row['case_id']] = info['source_hashes'] | info['cache_hashes']
            folders[partition].append(project/'data/stage3_cache_v1'/row['case_id'])
    if (out/'fingerprints.json').exists() and read(out/'fingerprints.json')!=fingerprints: raise ValueError('Data changed since run began')
    dump(out/'fingerprints.json',fingerprints)
    seed_all(args.seed); dev = device(); model = UNet3D(len(channels)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda',enabled=dev.type=='cuda'); history=[]; best=-1.; start_epoch=1
    if args.resume:
        ck = load_checkpoint(out/'last.pt'); model.load_state_dict(ck['model_state']); optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler']); history=ck['history']; best=ck['best']; start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['rng'])
        if dev.type=='cuda' and ck['cuda_rng']: torch.cuda.set_rng_state_all(ck['cuda_rng'])
    for epoch in range(start_epoch,args.epochs+1):
        began=time.perf_counter(); model.train(); losses=[]; optimizer.zero_grad(set_to_none=True)
        dataset=Patches(folders['train'],patch,channels,args.steps,args.seed,epoch)
        loader=DataLoader(dataset,batch_size=1,num_workers=0)
        for step,(x,y) in enumerate(loader,1):
            x,y=x.to(dev),y.to(dev)
            group_size=min(args.accumulate,args.steps-((step-1)//args.accumulate)*args.accumulate)
            with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda',dtype=torch.float16):
                loss=segmentation_loss(model(x),y)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            scaler.scale(loss/group_size).backward(); losses.append(float(loss.detach()))
            if step%args.accumulate==0 or step==args.steps:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step%25==0 or step==args.steps: print(f'Epoch {epoch}/{args.epochs} | patch {step}/{args.steps} | loss {np.mean(losses):.4f}',flush=True)
        entry={'epoch':epoch,'train_loss':float(np.mean(losses))}; validation=None
        if epoch%args.validate_every==0 or epoch==args.epochs:
            cases=[]
            for i,folder in enumerate(folders['validation'],1):
                print(f'Full-volume validation {i}/{len(folders["validation"])}',flush=True)
                prob=sliding(model,np.load(folder/'images.npy',mmap_mode='r'),dev,patch,channels)
                cases.append({'case_id':folder.name,'regions':region_metrics(prob.argmax(0),np.load(folder/'masks.npy',mmap_mode='r'),surface=False)})
            validation=summarize(cases); score=float(np.mean([r['mean_dice'] for r in validation.values() if r['mean_dice'] is not None]))
            entry['mean_regional_dice']=score; entry['validation']=validation
            dump(out/f'validation_epoch_{epoch:03d}.json',{'cases':cases,'summary':validation})
            if score>best:
                best=score
                save_checkpoint(out/'best.pt',{'model_state':model.state_dict(),'config':config,'epoch':epoch,
                    'validation':validation,'selection':'mean of defined WT/TC/ET case-mean Dice','data_fingerprint_sha256':sha(out/'fingerprints.json')})
                print(f'New best development Dice: {best:.4f}',flush=True)
        entry['seconds']=time.perf_counter()-began; history.append(entry); dump(out/'history.json',history)
        save_checkpoint(out/'last.pt',{'model_state':model.state_dict(),'config':config,'epoch':epoch,
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'history':history,'best':best,
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if dev.type=='cuda' else []})
    dump(out/'completion.json',{'best_validation_mean_regional_dice':best,'checkpoint_sha256':sha(out/'best.pt'),
        'internal_test_opened':False,'external_validation':False,'pilot':args.pilot})
    print(f'Completed: {out}\nDevelopment results only. Reserved test scans were not opened.')

def parser():
    p=argparse.ArgumentParser(description='True 3D development training. Never opens reserved internal-test images.'); p.add_argument('--project',default='.')
    p.add_argument('--output',default='runs/research_3d_pilot'); p.add_argument('--pilot',action='store_true')
    p.add_argument('--pilot-train',type=int,default=8); p.add_argument('--pilot-validation',type=int,default=2)
    p.add_argument('--epochs',type=int,default=3); p.add_argument('--steps',type=int,default=64)
    p.add_argument('--patch',type=int,nargs=3,default=[64,96,96]); p.add_argument('--channels',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--validate-every',type=int,default=1); p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke-only',action='store_true'); return p
if __name__=='__main__':
    args=parser().parse_args()
    if args.smoke_only: print(smoke(tuple(args.patch),len(args.channels)))
    else: train(args)

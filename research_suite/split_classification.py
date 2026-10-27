import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import rows,csv_out,dump,sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Create source/label-stratified, globally participant-disjoint classification splits.');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args();records=rows(args.input)
    out=Path(args.output)
    if out.exists():raise ValueError('Split file exists: preserve existing partitions')
    if not records:raise ValueError('No manifest rows')
    groups={};strata=defaultdict(list)
    for r in records:
        if r.get('split',''):raise ValueError('Input already has splits; do not repeatedly reshuffle')
        for key in ['participant_id','source','identity_evidence','label_evidence','protocol_evidence']:
            if not r.get(key):raise ValueError(f'Missing {key}')
        if r['label'] not in ['0','1']:raise ValueError('Label must be 0 or 1')
        k=r['participant_id'];sig=(r['source'],r['label'])
        if k in groups and groups[k]!=sig:raise ValueError('Cross-source/label identity conflict requires review')
        groups[k]=sig
    for k,sig in groups.items():strata[sig].append(k)
    assigned={};rng=np.random.default_rng(args.seed)
    for sig,ids in sorted(strata.items()):
        ids=sorted(ids);rng.shuffle(ids);n=len(ids)
        if n<10:raise ValueError(f'{sig}: only {n} participants; insufficient for separate train/validation/calibration/test partitions')
        nt=max(1,round(n*.15));nv=max(1,round(n*.15));nc=max(1,round(n*.1))
        for name,subset in [('test',ids[:nt]),('validation',ids[nt:nt+nv]),('calibration',ids[nt+nv:nt+nv+nc]),('train',ids[nt+nv+nc:])]:
            for k in subset:assigned[k]=name
    for r in records:r['split']=assigned[r['participant_id']]
    out.parent.mkdir(parents=True,exist_ok=True);csv_out(out,records)
    dump(out.with_suffix('.provenance.json'),{'source_manifest_sha256':sha(args.input),'seed':args.seed,
        'participant_counts':{s:sum(v==s for v in assigned.values()) for s in ['train','validation','calibration','test']},
        'note':'Grouping relies on verified global participant IDs; filenames alone are not identity evidence.'})
    print('Frozen classification splits saved:',out)

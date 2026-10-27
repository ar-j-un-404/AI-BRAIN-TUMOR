import csv
import hashlib
import json
from collections import Counter


def load_records(base):
    plan = json.loads((base/'btc_plan.json').read_text())
    with (base/'ixi_development.csv').open(newline='', encoding='utf-8') as f:
        ixi = list(csv.DictReader(f))
    lock = json.loads((base/'ixi_split_lock.json').read_text())
    expected = {i for i, role in lock['assignment'].items() if role == 'development'}
    if len(ixi) != 464 or {r['IXI_ID'] for r in ixi} != expected:
        raise ValueError('IXI development partition mismatch')
    btc = plan['manifest']
    ids = {r['participant_id'] for r in btc}
    if len(btc) != 36 or len(ids) != 36 or Counter(r['label'] for r in btc) != {0:11, 1:25}:
        raise ValueError('BTC manifest mismatch')
    seen = []
    for fold in plan['folds']:
        train, test = set(fold['train_ids']), set(fold['evaluation_ids'])
        if train & test or train | test != ids:
            raise ValueError('BTC participant leakage/incomplete fold')
        seen.extend(fold['evaluation_ids'])
    if Counter(seen) != Counter(ids) or sorted(f['fold'] for f in plan['folds']) != [1,2,3,4,5]:
        raise ValueError('BTC must evaluate each person exactly once')
    records = [dict(id=r['participant_id'], source='BTC', site='BTC', label=r['label'],
                    path=r['path'], expected_hash=r['decoded_array_sha256']) for r in btc]
    for site in sorted({r['site'] for r in ixi}):
        group = sorted([r for r in ixi if r['site']==site], key=lambda r:
                       hashlib.sha256(f"ixi-cv-v1|42|{site}|{r['IXI_ID']}".encode()).hexdigest())
        for i,r in enumerate(group):
            if r['role'] != 'development':
                raise ValueError('Reserved participant cannot enter this experiment')
            records.append(dict(id='IXI'+r['IXI_ID'].zfill(3), source='IXI', site=site,
                                label=0, path=r['path'], expected_hash=r['canonical_array_sha256'],
                                fold=i%5+1, metadata_review_required=r['metadata_review_required']))
    for r in records:
        if r['source']=='BTC':
            r['fold'] = next(f['fold'] for f in plan['folds'] if r['id'] in f['evaluation_ids'])
    return records


def metrics(rows):
    tn = sum(r['label']==0 and r['score']<.5 for r in rows)
    fp = sum(r['label']==0 and r['score']>=.5 for r in rows)
    fn = sum(r['label']==1 and r['score']<.5 for r in rows)
    tp = sum(r['label']==1 and r['score']>=.5 for r in rows)
    spec = tn/(tn+fp) if tn+fp else None
    sens = tp/(tp+fn) if tp+fn else None
    result = dict(n=len(rows), specificity=spec, false_positive_rate=1-spec if spec is not None else None,
                  confusion_matrix=[[tn,fp],[fn,tp]])
    if tn+fp and tp+fn:
        result.update(accuracy=(tn+tp)/len(rows), balanced_accuracy=(spec+sens)/2,
                      sensitivity=sens, precision=tp/(tp+fp) if tp+fp else 0,
                      f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return result

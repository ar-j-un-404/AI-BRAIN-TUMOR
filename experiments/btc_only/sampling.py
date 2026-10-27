import numpy as np

def make_schedule(records,fold,steps=1000):
    train=[r for r in records if r['fold']!=fold]
    healthy=[r for r in train if r['source']=='BTC' and r['label']==0]
    tumor=[r for r in train if r['source']=='BTC' and r['label']==1]
    sites=sorted({r['site'] for r in train if r['source']=='IXI'})
    pools={s:[r for r in train if r['source']=='IXI' and r['site']==s] for s in sites}
    rng=np.random.default_rng(42);replacement=np.random.default_rng(1042)
    schedule=[]
    for step in range(steps):
        chosen=[tumor[j] for j in rng.choice(len(tumor),2,replace=False)]
        first=healthy[int(rng.integers(len(healthy)))];chosen.append(first)
        site=sites[int(rng.integers(len(sites)))];ixi=pools[site][int(rng.integers(len(pools[site])))]
        order=rng.permutation(4)
        alternatives=[r for r in healthy if r['id']!=first['id']]
        extra=alternatives[int(replacement.integers(len(alternatives)))]
        original=chosen+[ixi];new=chosen+[extra]
        schedule.append(dict(pooled_ids=[original[j]['id'] for j in order],btc_ids=[new[j]['id'] for j in order]))
    return schedule

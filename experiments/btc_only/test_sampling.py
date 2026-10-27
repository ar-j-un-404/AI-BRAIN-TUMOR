import json
import unittest
from pathlib import Path
from collections import Counter
import numpy as np
from sampling import make_schedule

class Tests(unittest.TestCase):
    def test_fixed_folds_and_paired_schedule(self):
        records=json.loads((Path(__file__).parent/'reference.json').read_text())['source_experiment']['records']
        byid={r['id']:r for r in records}
        for fold in range(1,6):
            schedule=make_schedule(records,fold)
            self.assertEqual(schedule,make_schedule(records,fold))
            train=[r for r in records if r['fold']!=fold]
            healthy=[r for r in train if r['source']=='BTC' and r['label']==0]
            tumor=[r for r in train if r['source']=='BTC' and r['label']==1]
            sites=sorted({r['site'] for r in train if r['source']=='IXI'})
            pools={s:[r for r in train if r['source']=='IXI' and r['site']==s] for s in sites}
            rng=np.random.default_rng(42)
            for s in schedule:
                original=[tumor[j] for j in rng.choice(len(tumor),2,replace=False)]
                original+=[healthy[int(rng.integers(len(healthy)))]]
                site=sites[int(rng.integers(len(sites)))];original+=[pools[site][int(rng.integers(len(pools[site])))]]
                expected=[original[j]['id'] for j in rng.permutation(4)]
                self.assertEqual(s['pooled_ids'],expected)
                self.assertEqual(len(set(s['btc_ids'])),4)
                self.assertEqual(Counter(byid[i]['label'] for i in s['btc_ids']),{0:2,1:2})
                self.assertTrue(all(byid[i]['source']=='BTC' and byid[i]['fold']!=fold for i in s['btc_ids']))
                changes=[(a,b) for a,b in zip(s['pooled_ids'],s['btc_ids']) if a!=b]
                self.assertEqual(len(changes),1)
                self.assertEqual(byid[changes[0][0]]['source'],'IXI')
                self.assertEqual(byid[changes[0][1]]['label'],0)
if __name__=='__main__':unittest.main()

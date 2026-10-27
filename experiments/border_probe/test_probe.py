import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from features import features
from run_probe import load_cache,fit_predict

class Tests(unittest.TestCase):
    def test_center_is_excluded(self):
        rng=np.random.default_rng(42);a=rng.normal(size=(24,1,96,96)).astype('float32');b=a.copy()
        b[:,:,12:84,12:84]+=20
        np.testing.assert_array_equal(features(a,'border'),features(b,'border'))
        self.assertFalse(np.allclose(features(a,'full'),features(b,'full')))
        self.assertEqual(features(a,'border').shape,(168,))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):features(np.zeros((24,96,96)),'border')

    def test_cache_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cache').mkdir();p=root/'cache/test.npy'
            np.save(p,np.zeros((24,1,96,96),dtype='float32'))
            meta={'input':{'expected_hash':'reviewed'},'cache_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            (root/'cache/test.json').write_text(json.dumps(meta))
            r={'id':'test','expected_hash':'reviewed'};load_cache(root,r,False)
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):load_cache(root,r,False)

    def test_folds_and_fit(self):
        records=[]
        for fold in range(1,6):
            for source,label in [('BTC',0),('BTC',1),('IXI',0)]:
                records.append(dict(id=str(len(records)),source=source,site=source,fold=fold,label=label))
        X=np.random.default_rng(42).normal(size=(15,8))
        for task in ['pooled_disease','BTC_disease','healthy_source']:
            keep=[i for i,r in enumerate(records) if (r['source']=='BTC' if task=='BTC_disease' else r['label']==0 if task=='healthy_source' else True)]
            rr=[records[i] for i in keep]
            y=np.array([int(r['source']=='BTC') if task=='healthy_source' else r['label'] for r in rr])
            rows,checks=fit_predict(X[keep],y,np.array([r['fold'] for r in rr]),rr,task)
            self.assertEqual(len(rows),len(rr))
            for c in checks:self.assertFalse(set(c['train_ids'])&set(c['evaluation_ids']))

if __name__=='__main__':unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from review_core import load_case, panels, save_decision, sha

class Tests(unittest.TestCase):
    def test_geometry_hash_and_display(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); source=p/'mri.nii.gz'; mask=p/'mask.nii.gz'
            a=np.arange(8*10*12,dtype='float32').reshape(8,10,12)
            m=np.zeros(a.shape,dtype='uint8'); m[1:-1,2:-2,3:-3]=1
            aff=np.diag([2.,3.,4.,1.])
            nib.save(nib.Nifti1Image(a,aff),source); nib.save(nib.Nifti1Image(m,aff),mask)
            r=dict(mask_path=str(mask),input_sha256=sha(source),mask_sha256=sha(mask))
            g,b,s=load_case(r,dict(path=str(source)))
            self.assertTrue(np.array_equal(b,m.astype(bool)))
            for axis in range(3):
                imgs=panels(g,b,s,axis,4,.15)
                self.assertEqual(len(imgs),3)
                self.assertEqual(imgs[0].width,420)
            r['input_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'changed'): load_case(r,dict(path=str(source)))
            r['input_sha256']=sha(source)
            aff[0,3]=20; nib.save(nib.Nifti1Image(m,aff),mask); r['mask_sha256']=sha(mask)
            with self.assertRaisesRegex(ValueError,'geometry'): load_case(r,dict(path=str(source)))
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.json'
            save_decision(p,'a',{'status':'pending'})
            save_decision(p,'b',{'status':'needs_review'})
            save_decision(p,'a',{'status':'reject'})
            data=json.loads(p.read_text()); self.assertEqual(len(data),2)
            self.assertEqual(data['a']['status'],'reject')
if __name__=='__main__': unittest.main()

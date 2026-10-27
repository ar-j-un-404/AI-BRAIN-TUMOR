import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from audit_utils import sha,load_bag,statistics,montage

class Tests(unittest.TestCase):
    def test_cache_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.npy';a=np.zeros((24,1,96,96),dtype='float32');np.save(p,a)
            h=sha(p);np.testing.assert_array_equal(load_bag(p,h),a)
            np.save(p,a+1)
            with self.assertRaisesRegex(ValueError,'changed'):load_bag(p,h)
            np.save(p,np.zeros((2,2),dtype='float32'))
            with self.assertRaisesRegex(ValueError,'Invalid'):load_bag(p,sha(p))
    def test_statistics_and_display(self):
        a=np.ones((24,1,96,96),dtype='float32');b=a.copy();b[:3]=0
        s=statistics(a,b)
        self.assertEqual(s['extracted_nearly_empty_slices'],3)
        self.assertAlmostEqual(s['mean_absolute_change'],.125)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'preview.png';montage(a,b,p,'Synthetic input test')
            with Image.open(p) as image:self.assertEqual(image.size,(960,1120))
if __name__=='__main__':unittest.main()

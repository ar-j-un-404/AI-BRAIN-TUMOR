import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from geometry import physical_volume, prepare_physical, coverage, SPACING
from protocol import load_records

def phantom(spacing):
    n=int(120/spacing)
    x=(np.arange(n)-(n-1)/2)*spacing
    xx,yy,zz=np.meshgrid(x,x,x,indexing='ij')
    a=np.exp(-(xx**2+yy**2+zz**2)/(2*15**2)).astype('float32')
    affine=np.diag([spacing,spacing,spacing,1.])
    affine[:3,3]=-(n-1)*spacing/2
    image=nib.Nifti1Image(a,affine)
    image.header.set_xyzt_units('mm')
    return image,a

class Tests(unittest.TestCase):
    def test_physical_scale_and_center(self):
        a,meta=physical_volume(*phantom(1.))
        b,_=physical_volume(*phantom(2.))
        self.assertEqual(a.shape,(96,96,96))
        self.assertLess(float(np.abs(a-b).mean()),.001)
        affine=np.array(meta['target_affine'])
        np.testing.assert_allclose(nib.affines.apply_affine(affine,[47.5]*3),[0]*3)
        self.assertEqual(meta['target_spacing_mm'],[SPACING]*3)

    def test_orientation_and_shape(self):
        im,a=phantom(2.)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.nii.gz';nib.save(im,p)
            first,_,_=prepare_physical(p)
            flipped=nib.Nifti1Image(a[::-1],im.affine@np.array([[-1,0,0,a.shape[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
            flipped.header.set_xyzt_units('mm');nib.save(flipped,p)
            second,_,_=prepare_physical(p)
            self.assertEqual(first.shape,(24,1,96,96))
            np.testing.assert_allclose(first,second)
            with self.assertRaises(ValueError):prepare_physical(p,'invalid_hash')

    def test_no_silent_crop(self):
        im,a=phantom(2.)
        affine=im.affine.copy();affine[:3,:3]*=3
        oversized=nib.Nifti1Image(a,affine)
        oversized.header.set_xyzt_units('mm')
        with self.assertRaisesRegex(ValueError,'no silent crop'):
            physical_volume(oversized,a)

    def test_reported_failure_fits_new_grid(self):
        shape=(20,20,20)
        extent=np.array([112.00543964,144.91696939,138.12149137])
        affine=np.diag([*(2*extent/np.array(shape)),1.])
        image=nib.Nifti1Image(np.zeros(shape,dtype='float32'),affine)
        image.header.set_xyzt_units('mm')
        _,actual=coverage(image)
        np.testing.assert_allclose(actual,extent)
        self.assertTrue(np.any(actual>144))
        self.assertTrue(np.all(actual<160))

    def test_reviewed_splits(self):
        records=load_records(Path(__file__).resolve().parent)
        self.assertEqual(len(records),500)
        self.assertEqual(len({r['id'] for r in records}),500)
        self.assertEqual(sum(r['source']=='BTC' for r in records),36)
        for fold in range(1,6):
            train={r['id'] for r in records if r['fold']!=fold}
            test={r['id'] for r in records if r['fold']==fold}
            self.assertFalse(train&test)
            self.assertEqual(len(train|test),500)

if __name__=='__main__':unittest.main()

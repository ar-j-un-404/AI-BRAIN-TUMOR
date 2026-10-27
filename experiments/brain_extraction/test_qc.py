import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import json
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from qc import prepare_input,check_mask,select_cases,montage
from protocol import load_records

class Tests(unittest.TestCase):
    def test_controller_with_mock_inference_and_resume(self):
        import run_qc
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'assets').mkdir()
            (root/'assets/manifest.json').write_text(json.dumps({'model_folder':'synthetic-only'}))
            for name in ['run_qc.py','worker.py','qc.py','protocol.py']:(root/name).write_text('synthetic protocol')
            a=np.arange(20**3,dtype='float32').reshape(20,20,20)
            im=nib.Nifti1Image(a,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,root/'input.nii.gz')
            records=[dict(id=f'synthetic{i:02}',source='BTC',label=i%2,path=str(root/'input.nii.gz'),expected_hash=hashlib.sha256(a.tobytes()).hexdigest()) for i in range(45)]
            calls=[]
            def fake_run(cmd,**kwargs):
                calls.append(cmd)
                src=nib.load(cmd[cmd.index('--input')+1]);dst=Path(cmd[cmd.index('--output')+1])
                m=np.zeros(src.shape,dtype='uint8');m[3:-3,3:-3,3:-3]=1
                nib.save(nib.Nifti1Image(m,src.affine,src.header),dst)
                Path(str(dst)+'.device.json').write_text(json.dumps({'device':'cpu'}))
                return SimpleNamespace(returncode=0)
            with patch.object(run_qc,'BASE',root),patch.object(run_qc,'load_records',return_value=records),patch.object(run_qc,'select_cases',side_effect=lambda x:x),patch.object(run_qc.subprocess,'run',side_effect=fake_run),redirect_stdout(io.StringIO()):
                first=root/'report1';first.mkdir();self.assertEqual(run_qc.run(Path('mock-python'),first),0)
                self.assertEqual(len(calls),45)
                second=root/'report2';second.mkdir();self.assertEqual(run_qc.run(Path('mock-python'),second),0)
                self.assertEqual(len(calls),45)
                summary=json.loads((second/'summary.json').read_text())
                self.assertEqual(summary['visual_review_pending'],45)
                self.assertEqual(summary['failed'],0)
                self.assertFalse(summary['training_started'])

    def test_manifest(self):
        allrows=load_records(Path(__file__).resolve().parent);rows=select_cases(allrows)
        self.assertEqual(len(rows),45);self.assertEqual(len({r['id'] for r in rows}),45)
        self.assertEqual(sum(r['source']=='BTC' for r in rows),36)
        self.assertTrue({r['id'] for r in rows}<={r['id'] for r in allrows})

    def test_orientation_geometry_and_review(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);a=np.arange(24*26*28,dtype='float32').reshape(24,26,28)
            affine=np.diag([1.,1.2,1.5,1.]);im=nib.Nifti1Image(a,affine);im.header.set_xyzt_units('mm');nib.save(im,t/'in.nii.gz')
            r=dict(path=str(t/'in.nii.gz'),id='synthetic',source='BTC',expected_hash=hashlib.sha256(a.tobytes()).hexdigest())
            ras=prepare_input(r,t/'las.nii.gz');las=nib.load(t/'las.nii.gz')
            self.assertEqual(nib.aff2axcodes(las.affine),('L','A','S'))
            np.testing.assert_array_equal(nib.as_closest_canonical(las).get_fdata(),a)
            m=np.zeros(las.shape,dtype='uint8');m[3:-3,3:-3,3:-3]=1
            mask,stats=check_mask(ras,nib.Nifti1Image(m,las.affine))
            self.assertEqual(stats['visual_review'],'pending')
            montage(ras,mask,t/'preview.png','SYNTHETIC')
            self.assertTrue((t/'preview.png').is_file())
            with self.assertRaises(ValueError):check_mask(ras,nib.Nifti1Image(np.zeros_like(m),las.affine))
            bad=las.affine.copy();bad[0,3]+=5
            with self.assertRaisesRegex(ValueError,'geometry'):check_mask(ras,nib.Nifti1Image(m,bad))
            r['expected_hash']='wrong'
            with self.assertRaises(ValueError):prepare_input(r,t/'other.nii.gz')

if __name__=='__main__':unittest.main()

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np
import prepare
from protocol import load_records

BASE=Path(__file__).resolve().parent

class WorkflowTests(unittest.TestCase):
    def test_real_manifest_splits(self):
        records=load_records(BASE)
        self.assertEqual(len(records),500)
        self.assertEqual(len({r['id'] for r in records}),500)
        lock=json.loads((BASE/'ixi_split_lock.json').read_text())
        reserved={'IXI'+i.zfill(3) for i,role in lock['assignment'].items() if role!='development'}
        self.assertFalse(reserved & {r['id'] for r in records})
        for fold in range(1,6):
            train={r['id'] for r in records if r['fold']!=fold}
            test={r['id'] for r in records if r['fold']==fold}
            self.assertFalse(train & test)
            self.assertEqual(len(train|test),500)
            for label in [0,1]:self.assertTrue(any(r['source']=='BTC' and r['label']==label and r['fold']==fold for r in records))

    def test_preparation_reuse_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);old=base/'old';old.mkdir();out=base/'results';out.mkdir()
            (base/'settings.json').write_text(json.dumps({'extraction_folder':str(old),'worker_python':'fake-python'}))
            (base/'prior_review.json').write_text(json.dumps([{'id':'sub-PAT01','status':'needs_review'}]))
            records=[]
            for i in range(2):
                a=np.arange(16**3,dtype='float32').reshape(16,16,16)+i*100
                src=base/f'original{i}.nii.gz';im=nib.Nifti1Image(a,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,src)
                records.append(dict(id=['sub-PAT01','sub-CON01'][i],path=str(src),source='BTC',site='BTC',label=1-i,fold=i+1,
                    expected_hash=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()))
            mask=np.zeros((16,16,16),dtype='uint8');mask[1:-1,1:-1,1:-1]=1
            oldcase=old/'work/sub-PAT01';oldcase.mkdir(parents=True)
            raw=oldcase/'predicted_mask.nii.gz';nib.save(nib.Nifti1Image(mask,np.eye(4)),raw)
            (oldcase/'done.json').write_text(json.dumps({'key':{'protocol':'sig','input_sha256':prepare.sha(records[0]['path'])},'mask_sha256':prepare.sha(raw)}))
            calls=[]
            def fake_run(command,**kwargs):
                calls.append(command)
                dest=Path(command[command.index('--output')+1])
                las=nib.load(command[command.index('--input')+1])

                nib.save(nib.Nifti1Image(mask,las.affine),dest)
                return type('Result',(),{'returncode':0})()
            fake_bag=lambda a,m: np.ones((24,1,96,96),dtype='float32')*float(a[m].mean())
            with patch.object(prepare,'verify_assets',return_value=({},'sig')),patch.object(prepare.subprocess,'run',side_effect=fake_run),patch.object(prepare,'to_bag',side_effect=fake_bag),patch.object(prepare,'montage'):
                cache=prepare.prepare_all(base,out,records)
                self.assertEqual(len(calls),1)
                rows=json.loads((out/'extraction_cases.json').read_text())
                self.assertEqual(rows[0]['review_status'],'needs_review')
                self.assertEqual(rows[0]['extraction_origin'],'previous_extraction')
                prepare.prepare_all(base,out,records)
                self.assertEqual(len(calls),1)
                (cache/'sub-PAT01_brain.npy').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Cached training data changed'):
                    prepare.prepare_all(base,out,records)

                im=nib.load(records[0]['path']);arr=im.get_fdata(dtype=np.float32);arr[5,5,5]+=1
                nib.save(nib.Nifti1Image(arr,im.affine,im.header),records[0]['path'])
                with self.assertRaisesRegex(ValueError,'Input changed'):
                    prepare.prepare_all(base,out,records)

if __name__=='__main__':unittest.main()

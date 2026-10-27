import hashlib
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from contracts import select_folds,verify_fold,validate_experiment,sha,PREPROCESSING
from classifier_core import Classifier
from integrations import classify_t1,run_segment
from engine import prepare_inputs,mask_bytes

class ConnectionTests(unittest.TestCase):
    def test_unseen_and_known_fold_selection(self):
        rows=[{'source':'BTC','expected_hash':'abc','fold':3,'id':'person'}, {'source':'IXI','expected_hash':'def','fold':2,'id':'ixi'}]
        self.assertEqual(select_folds(rows,'abc','xyz'),([3],'person'))
        self.assertEqual(select_folds(rows,'xyz','def'),([2],'ixi'))
        self.assertEqual(select_folds(rows,'unseen','unseen'),([1,2,3,4,5],None))

    def test_classifier_end_to_end_and_fold_guard(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'synthetic_T1.nii.gz'
            a=np.random.default_rng(42).normal(size=(16,18,20)).astype(np.float32)
            image=nib.Nifti1Image(a,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,path)
            fingerprint=hashlib.sha256(a.tobytes()).hexdigest()
            records=[{'id':str(i),'source':'BTC' if i<36 else 'IXI','fold':i%5+1,'expected_hash':fingerprint if i==0 else f'unmatched-{i}','path':str(path) if i==0 else f'other-{i}.nii.gz'} for i in range(500)]
            exp={'version':'btc-ixi-paired-v1','preprocessing':PREPROCESSING,'steps':1000,'threshold':.5,'records':records,'source_hashes':{'core.py':sha(Path(__file__).resolve().parents[1]/'app/classifier_core.py')}}
            (root/'experiment.json').write_text(json.dumps(exp))
            model=Classifier()
            for p in model.parameters():p.data.zero_()
            model.head.bias.data.fill_(2)
            for f in range(1,6):
                folder=root/'BTC_plus_IXI'/f'fold_{f}';folder.mkdir(parents=True)
                protocol={'train_ids':[r['id'] for r in records if r['fold']!=f],'evaluation_ids':[r['id'] for r in records if r['fold']==f],'step':1000,'threshold':.5}
                (folder/'checkpoint_protocol.json').write_text(json.dumps(protocol))
                torch.save({'model':model.state_dict(),'step':1000},folder/'resume.pt')
            result=classify_t1(root,'BTC_plus_IXI',path)
            self.assertEqual(result['matched_development_participant'],'0');self.assertEqual(len(result['fold_scores']),1)
            self.assertAlmostEqual(result['score'],.880797,places=5)
            self.assertEqual(result['predicted_class'],'tumor')

            exp['records'][0]['expected_hash']='other';exp['records'][0]['path']='different.nii.gz'
            (root/'experiment.json').write_text(json.dumps(exp))
            result=classify_t1(root,'BTC_plus_IXI',path)
            self.assertEqual(len(result['fold_scores']),5);self.assertIsNone(result['matched_development_participant'])
            folder=root/'BTC_plus_IXI/fold_1';p=json.loads((folder/'checkpoint_protocol.json').read_text());p['train_ids'].append('0');(folder/'checkpoint_protocol.json').write_text(json.dumps(p))
            with self.assertRaises(ValueError):verify_fold(root,'BTC_plus_IXI',1,records)

    def test_segmentation_return_geometry_and_input_change_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'raw.nii.gz'
            data=np.random.default_rng(2).normal(100,20,size=(12,14,16,4)).astype(np.float32)
            aff=np.diag([-1.,1.,1.,1.]);aff[0,3]=11
            nii=nib.Nifti1Image(data,aff);nii.header.set_xyzt_units('mm');nib.save(nii,path)
            case=prepare_inputs([path]);pred=np.zeros(case['shape'],np.uint8);pred[2:4,3:5,4:6]=3

            output=root/'job';output.mkdir();p=output/'segmentation.nii.gz';p.write_bytes(mask_bytes(pred,case))
            (output/'worker_result.json').write_text(json.dumps({'prediction':str(p),'checkpoint_sha256':'mock'}))
            python=root/'.venv_nnunet/Scripts/python.exe';python.parent.mkdir(parents=True);python.touch()
            with patch('integrations.subprocess.Popen') as mock:
                mock.return_value.wait.return_value=0
                actual,info,extra=run_segment('nnU-Net 3D (reviewed final)',root,case,[path],output)
                np.testing.assert_array_equal(actual,pred)
            request=json.loads((output/'request.json').read_text());self.assertEqual(request['inputs'],[str(path)])
            data+=100;nib.save(nib.Nifti1Image(data,aff),path)
            with self.assertRaises(ValueError):run_segment('nnU-Net 3D (reviewed final)',root,case,[path],output)

if __name__=='__main__':unittest.main()

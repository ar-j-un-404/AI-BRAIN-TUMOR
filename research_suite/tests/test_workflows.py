from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import classify
from test_research import save_nii,manifest

def test_classifier_train_does_not_open_reserved_test(tmp_path,monkeypatch):

    monkeypatch.setattr(classify,'bootstrap_classification',lambda *a,**k:{})
    records=[];rng=np.random.default_rng(14)
    for split in ['train','validation','calibration','test']:
        for label in [0,1]:
            cid=f'{split}_{label}';t1=tmp_path/f'{cid}_t1.nii.gz';t2=tmp_path/f'{cid}_t2.nii.gz'
            if split!='test':
                save_nii(t1,rng.normal(size=(16,16,16)).astype(np.float32))
                save_nii(t2,rng.normal(size=(16,16,16)).astype(np.float32))
            records.append({'case_id':cid,'participant_id':cid,'source':'hospital','split':split,'label':str(label),'t1':str(t1),'t2':str(t2)})
    m=tmp_path/'manifest.csv';manifest(m,records);out=tmp_path/'classifier'
    classify.train(SimpleNamespace(manifest=str(m),output=str(out),seed=42,epochs=1,channels=[0,1]))
    saved=torch.load(out/'classifier.pt',weights_only=True)
    assert saved['temperature']>0 and saved['config']['input_indices']==[0,1]
    assert (out/'baseline_validation.json').is_file() and (out/'source_only_validation.json').is_file()

def test_app_initial_and_classifier_missing():
    from streamlit.testing.v1 import AppTest
    app=Path(__file__).resolve().parents[1]/'app.py';at=AppTest.from_file(str(app)).run(timeout=30)
    assert not at.exception
    at.sidebar.radio[0].set_value('Tumor-presence classifier').run()
    assert not at.exception and any('not trained yet' in x.value for x in at.info)

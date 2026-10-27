import hashlib
import json
from pathlib import Path

NN_HASH='ff740475f8d6ec14b49056fd5180d275da8cb3dca609061a5789dc208473d6a3'
PREPROCESSING='ras-zscore-clip5-cube96-24axial-mean-embedding'
MODEL_REL='nnunet_reference_v1/results/Dataset501_MSDReviewed/nnUNetTrainer_250epochs__nnUNetPlans_2GB__3d_fullres'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def verify_nnunet(folder):
    folder=Path(folder); here=Path(__file__).parent
    for actual,expected in [('dataset.json','expected_nnunet_dataset.json'),('plans.json','expected_nnunet_plans.json')]:
        if read(folder/actual)!=read(here/expected):raise ValueError('nnU-Net '+actual+' differs from the reviewed training configuration.')
    checkpoint=folder/'fold_0/checkpoint_final.pth'
    if sha(checkpoint)!=NN_HASH:raise ValueError('nnU-Net checkpoint does not match the reviewed 250-epoch final model.')
    return checkpoint

def validate_experiment(root):
    root=Path(root);exp=read(root/'experiment.json')
    if exp.get('version')!='btc-ixi-paired-v1' or exp.get('preprocessing')!=PREPROCESSING or exp.get('steps')!=1000 or exp.get('threshold')!=.5:
        raise ValueError('Unsupported classifier experiment/preprocessing.')
    current_hash=sha(Path(__file__).with_name('classifier_core.py'))
    accepted_hashes={current_hash}
    if current_hash=='9db3da39c37c58f07b9c58b5ec376a642772f37a04189a04095fbb3d08f204e0':
        accepted_hashes.add('ee8e78efe310e55f98eacb3e699abb80f91346d14ced4304f9c4d693db80de10')
        accepted_hashes.add('bc90080de5cbbde1b8be60acf11e229b662daaafe8213f5f4293092ed9adf7f0')
    if exp['source_hashes']['core.py'] not in accepted_hashes:
        raise ValueError('Classifier preprocessing code differs from training.')
    records=exp['records']
    if len(records)!=500 or len({r['id'] for r in records})!=500:raise ValueError('Classifier participant manifest is incomplete.')
    if any(r['fold'] not in range(1,6) for r in records):raise ValueError('Invalid fold assignment.')
    return exp

def select_folds(records, fingerprint, audit_fingerprint):
    matches=[r for r in records if r['expected_hash']==(audit_fingerprint if r['source']=='IXI' else fingerprint)]
    if len(matches)>1:raise ValueError('Multiple identities match this MRI; resolve before inference.')
    return ([matches[0]['fold']],matches[0]['id']) if matches else ([1,2,3,4,5],None)

def verify_fold(root, variant, fold, records):
    if variant not in ('BTC_only','BTC_plus_IXI'):raise ValueError('Unknown classifier variant.')
    folder=Path(root)/variant/f'fold_{fold}'
    protocol=read(folder/'checkpoint_protocol.json')
    expected_train={r['id'] for r in records if r['fold']!=fold and (variant=='BTC_plus_IXI' or r['source']=='BTC')}
    expected_eval={r['id'] for r in records if r['fold']==fold}
    if set(protocol['train_ids'])!=expected_train or set(protocol['evaluation_ids'])!=expected_eval or expected_train&expected_eval:
        raise ValueError('Classifier fold membership mismatch.')
    if protocol.get('step')!=1000 or protocol.get('threshold')!=.5:raise ValueError('Incomplete classifier fold.')
    checkpoint=folder/'resume.pt'
    if not checkpoint.is_file():raise FileNotFoundError('Trained classifier checkpoint missing: '+str(checkpoint))
    return checkpoint

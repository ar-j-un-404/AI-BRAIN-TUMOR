import json
import shutil
import uuid
from pathlib import Path
import numpy as np
import streamlit as st
from contracts import MODEL_REL,read
from engine import prepare_inputs,mask_bytes,measurements
from viewer import build_viewer
from app_support import analysis_bundle,report_payload
from integrations import run_segment,classify_t1

st.set_page_config(page_title='BrainTumor AI · Connected',layout='wide')
HERE=Path(__file__).resolve().parent
if not (HERE/'settings.json').is_file():
    st.error('Run python launch.py configure before opening the app.');st.stop()
SETTINGS=read(HERE/'settings.json')
project=Path(SETTINGS['project'])
st.title('BrainTumor AI')
st.caption('Connected research workspace · Separate 2D and 3D views · Model provenance recorded')
page=st.sidebar.radio('Workspace',['MRI segmentation & 3D','T1 classification','Models & results','Experiment comparisons'])

if page=='MRI segmentation & 3D':
    st.info('Use aligned, preprocessed, skull-stripped MRI: FLAIR → T1w → T1ce → T2w. The app does not register or skull-strip inputs.')
    source=st.radio('MRI input',['Local four-channel NIfTI','Four assigned local modalities','Upload four-channel NIfTI'],horizontal=True)
    paths=[];upload=None
    if source=='Local four-channel NIfTI':
        value=st.text_input('MRI path');paths=[value] if value.strip() else []
    elif source=='Four assigned local modalities':
        paths=[st.text_input(name+' path',key='path_'+name) for name in ['FLAIR','T1w','T1ce','T2w']]
    else:upload=st.file_uploader('MRI NIfTI (.nii / .nii.gz)',type=['nii','gz'])
    ref=st.text_input('Optional reference mask path (MSD labels 0,1,2,3)')
    confirmed=st.checkbox('I verified modality order, alignment and preprocessing.')
    if st.button('Load MRI',disabled=not confirmed):
        st.session_state.pop('seg_case',None);st.session_state.pop('seg_paths',None)
        st.session_state.seg_results={};st.session_state.pop('seg_bundle',None)
        old=st.session_state.pop('upload_dir',None)
        if old:shutil.rmtree(old,ignore_errors=True)
        try:
            if upload is not None:
                folder=HERE/'sessions'/uuid.uuid4().hex;folder.mkdir(parents=True)
                suffix='.nii.gz' if upload.name.lower().endswith('.gz') else '.nii'
                f=folder/('input'+suffix);f.write_bytes(upload.getvalue());paths=[str(f)]
                st.session_state.upload_dir=str(folder)
            if not paths or any(not p.strip() for p in paths):raise ValueError('Supply the MRI input(s).')
            paths=[str(Path(p.strip().strip('"')).expanduser().resolve()) for p in paths]
            reference=ref.strip().strip('"') or None
            with st.spinner('Checking MRI geometry and preparing display…'):case=prepare_inputs(paths,reference)
            st.session_state.seg_case=case;st.session_state.seg_paths=paths
            st.session_state.seg_description=upload.name if upload is not None else paths[0]
        except Exception as e:st.error(str(e))
    case=st.session_state.get('seg_case')
    if case is not None:
        st.caption('Loaded scan: '+st.session_state.seg_description+' · Changing a path requires Load MRI again.')
        choices=['nnU-Net 3D (reviewed final)','Existing 2D segmentation','Small 3D segmentation']
        selection=st.selectbox('Segmentation model',choices)
        mc=st.checkbox('Small 3D: add four-pass MC uncertainty',disabled=selection!='Small 3D segmentation')
        c1,c2=st.columns(2)
        single=c1.button('Run selected segmentation',type='primary')
        all_models=c2.button('Compare all three segmentation models')
        if single or all_models:
            jobs=choices if all_models else [selection]
            for kind in jobs:

                st.session_state.seg_results.pop(kind,None);st.session_state.pop('seg_bundle',None)
                output=HERE/'jobs'/uuid.uuid4().hex
                try:
                    with st.spinner('Running '+kind+' — progress details are in the terminal/job log…'):
                        pred,info,extra=run_segment(kind,project,case,st.session_state.seg_paths,output,mc=mc and kind=='Small 3D segmentation')
                    st.session_state.seg_results[kind]={'pred':pred,'info':info,'extra':extra,'job':str(output)}
                    st.success(kind+' finished.')
                except Exception as e:st.error(kind+': '+str(e))
        results=st.session_state.seg_results
        if results:
            active=st.selectbox('Result to display',list(results))
            result=results[active];pred=result['pred'];info=result['info']
            st.caption('Displayed prediction: '+active+' · checkpoint '+info['checkpoint_sha256'][:16])
        else:pred=None;result=None;info=None
        modality=st.selectbox('Display modality',range(4),format_func=lambda i:['FLAIR','T1w','T1ce','T2w'][i])
        try:st.iframe(build_viewer(case,modality,pred),height=820)
        except Exception as e:st.error('Viewer: '+str(e))
        if result:
            st.write('Model marked tumor-region voxels.' if np.any(pred) else 'Model marked no tumor-region voxels; this does not establish a healthy examination.')
            st.dataframe(measurements(pred,case),hide_index=True)
            st.caption('Dice is overlap with this supplied reference mask, when available. It is not tumor-presence classification accuracy.')
            st.download_button('Download predicted mask',mask_bytes(pred,case),'segmentation_native.nii.gz')
            payload=report_payload(case,pred,info)
            st.download_button('Download model provenance & measurements',json.dumps(payload,indent=2),'segmentation_analysis.json')
            if st.button('Prepare case report ZIP'):
                extra=result['extra'];st.session_state.seg_bundle=(active,analysis_bundle(case,pred,info,extra.get('entropy'),extra.get('mi')))
            bundle=st.session_state.get('seg_bundle')
            if bundle and bundle[0]==active:st.download_button('Download case report ZIP',bundle[1],'brain_tumor_case_report.zip')
            with st.expander('Checkpoint and preprocessing details'):st.json(info)
            extra=result['extra']
            if 'entropy' in extra:
                z=st.slider('Uncertainty axial slice',0,case['shape'][2]-1,case['shape'][2]//2)
                st.image(extra['entropy'][:,:,z].T[::-1],caption='Small 3D predictive entropy — not calibrated confidence',clamp=True)
        if len(results)>1:
            rows=[]
            for name,r in results.items():
                for row in measurements(r['pred'],case):rows.append({'model':name,**row})
            st.subheader('Same-scan segmentation comparison');st.dataframe(rows,hide_index=True)
        if st.button('Clear loaded MRI and viewer results'):
            for key in ['seg_case','seg_paths','seg_results','seg_bundle']:st.session_state.pop(key,None)
            old=st.session_state.pop('upload_dir',None)
            if old:shutil.rmtree(old,ignore_errors=True)
            st.rerun()

elif page=='T1 classification':
    st.subheader('Tumor-presence classifier · exploratory')
    st.warning('BTC + IXI development results: 75% BTC accuracy, 59.09% balanced accuracy; 9/11 BTC healthy participants were false positives. This classifier is not a reliable diagnosis.')
    root=st.text_input('BTC_IXI_Training results folder',SETTINGS.get('classifier_results',''))
    variant=st.selectbox('Classifier',['BTC_plus_IXI','BTC_only'])
    t1=st.text_input('Single 3D T1w MRI path')
    confirm=st.checkbox('This is a T1w research scan with reviewed provenance, not a reserved test participant.')
    signature=(root,variant,t1)
    if st.session_state.get('classification_selection')!=signature:
        st.session_state.pop('classification_result',None);st.session_state.classification_selection=signature
    if st.button('Run T1 classification',disabled=not confirm or not root.strip() or not t1.strip()):
        st.session_state.pop('classification_result',None)
        try:
            with st.spinner('Checking fold membership and running the T1 classifier…'):
                st.session_state.classification_result=classify_t1(Path(root.strip().strip('"')),variant,Path(t1.strip().strip('"')))
        except Exception as e:st.error(str(e))
    r=st.session_state.get('classification_result')
    if r:
        st.subheader('Research prediction: '+r['predicted_class'].upper())
        st.write('Model score:',round(r['score'],4));st.caption('Threshold 0.5. This score is not a calibrated probability or accuracy.')
        st.info(r['evaluation_context']);st.dataframe(r['fold_scores'],hide_index=True)
        st.download_button('Download classification report',json.dumps(r,indent=2),'classification_research.json')
        with st.expander('Provenance'):st.json(r)
    st.caption('Known development scans use only their held-out fold. Unmatched scans use the mean of five fold scores, an unvalidated ensemble. These settings do not create new performance evidence.')

elif page=='Models & results':
    st.subheader('Connected models')
    checks={
        'nnU-Net final checkpoint':project/MODEL_REL/'fold_0/checkpoint_final.pth',
        'nnU-Net Python':project/'.venv_nnunet/Scripts/python.exe',
        'Frozen 2D checkpoint':project/'runs/stage6_internal_test_001/frozen_model.pt',
        'Stage 4 2D fallback':project/'runs/stage4_regions_001/best_model.pt',
        'Stage 3 2D fallback':project/'runs/stage3_full_001/best_model.pt',
        'Small 3D checkpoint':project/'runs/3d_full_all4_seed42/best.pt'}
    root=Path(SETTINGS.get('classifier_results') or 'CLASSIFIER_NOT_CONFIGURED')
    for variant in ['BTC_plus_IXI','BTC_only']:
        for fold in range(1,6):checks[f'{variant} fold {fold}']=root/variant/f'fold_{fold}/resume.pt'
    st.dataframe([{'component':n,'file_exists':p.is_file(),'path':str(p)} for n,p in checks.items()],hide_index=True)
    st.caption('Existence is not validation: checkpoint hashes, metadata and fold membership are checked during inference. Missing optional models do not block other tasks.')
    st.subheader('Previously reported results')
    st.table([{'model':'nnU-Net','evaluation':'73 development validation cases','WT Dice':'90.83%','TC Dice':'83.41%','ET Dice':'76.76%'},
              {'model':'Small 3D','evaluation':'73 development validation cases','WT Dice':'82.38%','TC Dice':'66.86%','ET Dice':'65.12%'}])
    st.write('BTC + IXI classifier: BTC accuracy 75%, balanced accuracy 59.09%, tumor sensitivity 100% (25/25), BTC healthy specificity 18.18% (2/11). IXI development healthy specificity 100% (464/464). These reused development results are not untouched external validation.')
    st.caption('The 117 reserved IXI scans and segmentation test batches are not automatically evaluated. Single-scan demo results do not replace a frozen evaluation protocol.')
    st.write('The previous app remains available through your original launcher. This connected app adds inference routes; it does not retrain or replace model weights.')

else:
    st.subheader('Experiment comparisons')
    st.caption('Previously reported results. This page does not run training or evaluate new scans.')
    results=read(HERE/'results_summary.json')
    st.write('Segmentation: overlap scores, not tumor-versus-healthy accuracy')
    st.dataframe(results['segmentation'],hide_index=True)
    st.write('Classification: BTC development participants')
    st.dataframe(results['classification'],hide_index=True)
    st.warning('These development experiments reuse participants. Brain extraction and physical-space variants did not improve classification and are not installed as replacement models.')
    st.info('External validation and reliable tumor-versus-healthy classification remain unfinished. IXI contains healthy controls only; its specificity is not two-class accuracy.')
    st.download_button('Download aggregate results',json.dumps(results,indent=2),'results_summary.json')

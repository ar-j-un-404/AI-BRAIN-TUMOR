import io
import json
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import streamlit as st
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from scipy.special import expit
from common import read,sha,load_checkpoint,device
from engine import prepare_inputs,mask_bytes,measurements,load_frozen,infer
from viewer import build_viewer
from networks import UNet3D,Classifier3D
from volumes import sliding,uncertainty_maps
from classify import load_pair
from explain import gradcam

st.set_page_config(page_title='BrainTumor AI Research',layout='wide')
st.title('BrainTumor AI · Research workspace')
st.caption('Segmentation, tumor-presence experiments and reliability review. Research use only.')
project=Path(st.sidebar.text_input('Project folder',str(Path.cwd()))).resolve()
page=st.sidebar.radio('Workspace',['MRI segmentation','Tumor-presence classifier','Reports and limitations'])

def mesh_figure(pred,affine):
    fig=go.Figure()
    for value,name,color in [(1,'Edema','#ef963c'),(2,'Non-enhancing tumor','#20b6b0'),(3,'Enhancing tumor','#b47bf0')]:
        binary=pred[::2,::2,::2]==value
        if not binary.any(): continue

        verts,faces,_,_=marching_cubes(np.pad(binary.astype(np.float32),1),.5)
        verts=(verts-1)*2; world=nib.affines.apply_affine(affine,verts)
        fig.add_trace(go.Mesh3d(x=world[:,0],y=world[:,1],z=world[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
            color=color,opacity=.65,name=name,showlegend=True))
    fig.update_layout(scene=dict(aspectmode='data',xaxis_title='R (mm)',yaxis_title='A (mm)',zaxis_title='S (mm)'),height=620,margin=dict(l=0,r=0,b=0,t=0))
    return fig

if page=='MRI segmentation':
    mode=st.selectbox('Segmentation model',['Existing frozen Stage 6 2D model','New research-suite 3D checkpoint'])
    checkpoint=st.text_input('3D checkpoint path',str(project/'runs/3d_pilot_all4_seed42/best.pt')) if mode.startswith('New') else ''
    st.info('Input must be preprocessed, aligned and skull-stripped. Four-channel order: FLAIR, T1w, T1ce, T2w. This app does not register or skull-strip scans.')
    source=st.radio('Input method',['Local four-channel NIfTI','Upload four-channel NIfTI','Upload four assigned modalities'])
    local=st.text_input('MRI path') if source.startswith('Local') else ''
    uploads=[]
    if source=='Upload four-channel NIfTI':
        upload=st.file_uploader('Four-channel volume',type=['nii','gz']); uploads=[upload] if upload else []
    elif source=='Upload four assigned modalities':
        uploads=[st.file_uploader(name,type=['nii','gz'],key=name) for name in ['FLAIR','T1w','T1ce','T2w']]
    reference=st.text_input('Optional reference mask path (MSD labels 0,1,2,3)')
    confirmed=st.checkbox('I verified modality order and the required preprocessing.')
    if st.button('Load MRI',disabled=not confirmed):

        for k in ['case','prediction','entropy','mi','model_info']: st.session_state.pop(k,None)
        try:
            with tempfile.TemporaryDirectory() as td:
                if source.startswith('Local'): paths=[local]
                else:
                    if not uploads or any(u is None for u in uploads): raise ValueError('Select all required input files')
                    paths=[]
                    for i,u in enumerate(uploads):
                        suffix='.nii.gz' if u.name.lower().endswith('.gz') else '.nii'
                        path=Path(td)/f'input_{i}{suffix}'; path.write_bytes(u.getvalue()); paths.append(path)
                st.session_state.case=prepare_inputs(paths,reference or None)
        except Exception as e: st.error(str(e))
    case=st.session_state.get('case')
    if case is not None:
        st.write({'canonical_shape':case['shape'],'spacing_mm':case['spacing'].tolist(),'modalities':['FLAIR','T1w','T1ce','T2w']})
        mc=st.checkbox('Add 4-pass MC-dropout uncertainty (3D only; several times slower)',disabled=not mode.startswith('New'))
        if st.button('Run segmentation'):
            for k in ['prediction','entropy','mi','model_info']: st.session_state.pop(k,None)
            try:
                with st.spinner('Running volume inference…'):
                    dev=device()
                    if mode.startswith('New'):
                        ck=load_checkpoint(checkpoint); config=ck['config']
                        if config['task']!='segmentation3d' or config['protocol']!='msd-normalized-native-1mm-3d-v1': raise ValueError('Unsupported 3D protocol')
                        if not np.allclose(case['spacing'],[1,1,1],atol=1e-3): raise ValueError('3D model requires reviewed 1-mm isotropic input')
                        model=UNet3D(len(config['channels'])).to(dev); model.load_state_dict(ck['model_state'])
                        prob=sliding(model,case['images'],dev,config['patch'],config['channels'])
                        pred=prob.argmax(0).astype(np.uint8).transpose(2,1,0)
                        st.session_state.entropy=(-(prob*np.log(np.maximum(prob,1e-8))).sum(0)/np.log(4)).transpose(2,1,0)
                        if mc:
                            _,_,mi=uncertainty_maps(model,case['images'],dev,config['patch'],config['channels'],4)
                            st.session_state.mi=mi.transpose(2,1,0)
                        model_info={'kind':'3D research model','checkpoint_sha256':sha(checkpoint),'epoch':ck['epoch'],'pilot':config['pilot']}
                    else:
                        model,plan=load_frozen(project,dev); pred=infer(model,case,dev)
                        model_info={'kind':'frozen 2D model','checkpoint_sha256':plan['checkpoint_sha256']}
                    st.session_state.prediction=pred; st.session_state.model_info=model_info
            except Exception as e: st.error(str(e))
        pred=st.session_state.get('prediction'); modality=st.selectbox('Display modality',range(4),format_func=lambda i:['FLAIR','T1w','T1ce','T2w'][i])
        st.iframe(build_viewer(case,modality,pred),height=820)
        if pred is not None:
            st.caption(str(st.session_state.model_info))
            st.write('Model marked tumor-region voxels.' if np.any(pred) else 'Model marked no tumor-region voxels. This does not establish a tumor-free examination.')
            st.dataframe(measurements(pred,case),hide_index=True)
            st.download_button('Download native-grid segmentation',mask_bytes(pred,case),'segmentation_native.nii.gz')
            st.download_button('Download analysis JSON',json.dumps({'model':st.session_state.model_info,'input_sha256':case['input_sha256'],'measurements':measurements(pred,case)},indent=2),'analysis.json')
            with st.expander('Interactive 3D tumor surfaces',expanded=False):
                if np.any(pred): st.plotly_chart(mesh_figure(pred,case['affine']),use_container_width=True)
                else: st.info('No non-background surface to render.')
                st.caption('Surfaces downsampled for display only; exported segmentation remains full resolution.')
            if 'entropy' in st.session_state:
                z=st.slider('Uncertainty axial slice',0,case['shape'][2]-1,case['shape'][2]//2)
                cols=st.columns(2); cols[0].image(st.session_state.entropy[:,:,z].T[::-1],caption='Normalized predictive entropy',clamp=True)
                if 'mi' in st.session_state: cols[1].image(st.session_state.mi[:,:,z].T[::-1],caption='MC-dropout mutual information',clamp=True)
                st.caption('Uncertainty values are not calibrated disease probabilities or a validated referral rule.')
                image=nib.Nifti1Image(st.session_state.entropy.astype(np.float32),case['affine']); image.header.set_xyzt_units('mm')
                st.download_button('Download canonical-grid entropy NIfTI',image.to_bytes(),'entropy_canonical.nii')
elif page=='Tumor-presence classifier':
    st.write('This is a separate T1/T2 classification model. Segmentation voxels are not used as a tumor/no-tumor diagnosis.')
    ckpath=st.text_input('Classifier checkpoint',str(project/'runs/classification_001/classifier.pt'))
    if not Path(ckpath).is_file(): st.info('Classifier not trained yet. Supply a verified two-class participant manifest and run the classification workflow. No tumor/no-tumor result is available from the existing segmentation model.')
    t1=st.text_input('Registered, skull-stripped T1w NIfTI'); t2=st.text_input('Registered, skull-stripped T2w NIfTI')
    confirm=st.checkbox('T1/T2 preprocessing matches the classifier training protocol.')
    if st.button('Classify examination',disabled=not confirm or not Path(ckpath).is_file()):
        try:
            ck=load_checkpoint(ckpath)
            if ck['config']['task']!='classification3d': raise ValueError('Wrong model task')
            model=Classifier3D(len(ck['config']['input_indices'])).to(device()); model.load_state_dict(ck['model_state']); model.eval()
            x=load_pair({'t1':t1,'t2':t2})[ck['config']['input_indices']][None].to(device())
            with torch.inference_mode(): logit=float(model(x)[0])
            score=float(expit(logit/ck['temperature'])); positive=score>=ck['threshold']
            st.subheader('Research prediction: tumor-positive' if positive else 'Research prediction: tumor-negative')
            st.metric('Temperature-scaled model score',f'{score:.3f}')
            st.caption('Not a diagnosis. Calibration was fitted on development participants; validity on new cohorts must be measured separately.')
            cam=gradcam(model,x); z=32
            st.image(cam[z],caption='Positive-class Grad-CAM, central slice of resized model grid',clamp=True)
            st.download_button('Download classification result',json.dumps({'prediction':int(positive),'score':score,'threshold':ck['threshold'],'checkpoint_sha256':sha(ckpath),'research_use_only':True},indent=2),'classification.json')
        except Exception as e: st.error(str(e))
else:
    st.write('Existing Stage 4 development validation: WT 89.62%, TC 78.95%, ET 76.79% Dice. These are segmentation overlap metrics, not classification accuracy.')
    st.write('New 3D/classifier/external scores are unavailable until their workflows run. The application does not manufacture missing results.')
    report=st.text_input('Report JSON path',str(project/'runs/stage6_internal_test_001/stage6_summary.json'))
    if Path(report).is_file():
        try: st.json(read(report))
        except Exception as e: st.error(str(e))
    else: st.info('No report at this path.')
    st.markdown('[Project methods and citations](https://github.com/MIC-DKFZ/nnUNet) · Read docs/RESEARCH_REPORT.md and docs/DATA_AND_LITERATURE.md in the package.')

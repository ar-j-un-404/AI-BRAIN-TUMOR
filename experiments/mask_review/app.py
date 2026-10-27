import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from review_core import load_case, panels, save_decision, sha

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Brain mask review', layout='wide')
st.title('Brain extraction • full-volume review')
st.caption('Review preparation for classifier experiments. This tool does not diagnose tumors or alter MRI files.')
settings = BASE/'settings.json'
initial = json.loads(settings.read_text()).get('report', '') if settings.exists() else ''
report_text = st.sidebar.text_input('Extraction report folder', initial)
report = Path(report_text.strip().strip('"'))
if not (report/'cases.json').is_file() or not (report/'selected_cases.json').is_file():
    st.info('Choose the extracted report folder containing cases.json and selected_cases.json. Original MRI and work/mask files must remain on this computer.')
    st.stop()
settings.write_text(json.dumps({'report': str(report)}))
records = json.loads((report/'cases.json').read_text())
selected = {r['id']: r for r in json.loads((report/'selected_cases.json').read_text())}
records = [r for r in records if r.get('status') == 'visual_review_pending']
if not records:
    st.error('No successful extraction cases in this report.'); st.stop()

key = sha(report/'cases.json')[:16]
output = BASE/'reviews'/key; output.mkdir(parents=True, exist_ok=True)
review_path = output/'decisions.json'
reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
ids = [r['id'] for r in records]
case_id = st.sidebar.selectbox('Case', ids, format_func=lambda i: f"{i} — {reviews.get(i, {}).get('status', 'pending')}")
r = records[ids.index(case_id)]
st.subheader(f"{case_id} · {r['source']}")
st.write('Automatic flags:', r.get('automatic_flags') or 'None reported')

@st.cache_data(max_entries=2, show_spinner=False)
def cached(record_json, selected_json, source_stat, mask_stat):
    return load_case(json.loads(record_json), json.loads(selected_json))

try:
    src = Path(selected[case_id]['path']); mask = Path(r['mask_path'])
    with st.spinner('Loading and verifying this MRI and mask…'):
        gray, binary, spacing = cached(json.dumps(r, sort_keys=True), json.dumps(selected[case_id], sort_keys=True),
                                      (src.stat().st_size, src.stat().st_mtime_ns),
                                      (mask.stat().st_size, mask.stat().st_mtime_ns))
except Exception as exc:
    st.error(str(exc)); st.stop()
axis = st.sidebar.radio('Plane', [2, 1, 0], format_func=lambda a: {2:'Axial', 1:'Coronal', 0:'Sagittal'}[a])
index = st.slider('Slice (zero-based)', 0, gray.shape[axis]-1, gray.shape[axis]//2, key=f'{case_id}_{axis}')
opacity = st.sidebar.slider('Mask fill opacity', 0.0, 0.6, 0.15, 0.05)
for col, label, image in zip(st.columns(3), ['Original MRI', 'Green mask / yellow boundary', 'Extracted brain'], panels(gray,binary,spacing,axis,index,opacity)):
    with col:
        st.markdown('**'+label+'**'); st.image(image, use_container_width=True)
st.caption('Canonical RAS, physical aspect ratio preserved. Axial: left=Left, right=Right, top=Anterior. Coronal: left=Left, right=Right, top=Superior. Sagittal: left=Posterior, right=Anterior, top=Superior.')
st.info('Inspect the full slice range in all three planes. Check for missing brain or visible lesion tissue and retained skull/scalp. If uncertain, choose Needs expert review. These are visual QC judgments, not validated tumor-preservation measurements.')
with st.form(f'decision_{case_id}'):
    reviewer = st.text_input('Reviewer name or initials')
    inspected = st.checkbox('I inspected the full volume in all three planes')
    boundary = st.selectbox('Brain boundary', ['Not assessed', 'Acceptable', 'Needs expert review', 'Defect observed'])
    preservation = st.selectbox('Visible lesion tissue', ['Not assessed', 'Appears preserved', 'Needs expert review', 'Possible tissue removed', 'Not applicable — control'])
    status = st.selectbox('Decision', ['pending', 'needs_review', 'reject', 'accept_visual_qc'])
    notes = st.text_area('Notes (location and slice number for any concern)')
    submit = st.form_submit_button('Save review decision')
    if submit:
        tumor = selected[case_id].get('label') == 1
        acceptable_lesion = preservation == 'Appears preserved' if tumor else preservation in ['Appears preserved', 'Not applicable — control']
        if not reviewer.strip():
            st.error('Enter reviewer initials to save.')
        elif status == 'accept_visual_qc' and not (inspected and boundary == 'Acceptable' and acceptable_lesion):
            st.error('Acceptance requires full-volume inspection, acceptable boundary, and an appropriate lesion-tissue assessment.')
        elif status in ['reject','needs_review'] and not notes.strip():
            st.error('Describe the concern or uncertainty in Notes.')
        else:
            save_decision(review_path, case_id, dict(status=status, reviewer=reviewer.strip(), full_volume_inspected=inspected,
                          boundary=boundary, lesion_assessment=preservation, notes=notes,
                          input_sha256=r['input_sha256'], mask_sha256=r['mask_sha256'],
                          timestamp_utc=datetime.now(timezone.utc).isoformat()))
            st.rerun()
st.sidebar.write(f"Saved decisions: {len(reviews)}/{len(ids)}")
st.sidebar.write(f"Accepted visual QC: {sum(d.get('status')=='accept_visual_qc' for d in reviews.values())}/{len(ids)}")

rows = [dict(id=i, **reviews.get(i, {'status':'pending'})) for i in ids]
stream = io.StringIO(); fields = sorted(set().union(*(row.keys() for row in rows)))
writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('review.csv', stream.getvalue())
    z.writestr('decisions.json', json.dumps(rows, indent=2))
    z.writestr('provenance.json', json.dumps({'extraction_report_sha256':sha(report/'cases.json'), 'training_started':False,
                 'note':'Human visual QC only. Pending and uncertain cases are retained for review, not automatically excluded.'},indent=2))
st.sidebar.download_button('Download review ZIP', buffer.getvalue(), 'brain_mask_review.zip', 'application/zip')
st.caption('Decisions save locally in this review folder. Existing extraction reports and model files stay unchanged.')

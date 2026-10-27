import csv
import gzip
import html
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from engine import PROTOCOL, SmallUNet, measurements, mask_bytes, sha256, load_frozen
from common import load_checkpoint


def project_path(project, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project) / path).resolve()


def load_2d(project, dev):
    
    frozen = Path(project) / 'runs/stage6_internal_test_001'
    if (frozen / 'evaluation_plan.json').exists():
        model, plan = load_frozen(project, dev)
        return model, {'kind': 'Frozen 2D model', 'checkpoint_sha256': plan['checkpoint_sha256'],
                       'checkpoint': str(frozen / 'frozen_model.pt')}
    path = Path(project) / 'runs/stage4_regions_001/best_model.pt'
    if not path.is_file():
        raise FileNotFoundError('No frozen Stage 6 model or Stage 4 best_model.pt found. Keep your existing runs folder beside research_suite.')
    ck = load_checkpoint(path)
    if ck['configuration']['protocol'] != PROTOCOL:
        raise ValueError('Unsupported 2D preprocessing protocol')
    model = SmallUNet().to(dev)
    model.load_state_dict(ck['model_state']); model.eval()
    return model, {'kind': 'Stage 4 development 2D model', 'checkpoint_sha256': sha256(path),
                   'checkpoint': str(path), 'epoch': ck.get('epoch')}


def report_payload(case, pred, model_info):
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'research_use_only': True,
            'result': 'Tumor-region voxels marked' if np.any(pred) else 'No tumor-region voxels marked',
            'interpretation': 'Segmentation output is not a tumor/no-tumor diagnosis. An empty mask does not establish a healthy scan.',
            'model': model_info, 'input_sha256': case['input_sha256'],
            'reference_sha256': case.get('reference_sha256'), 'protocol': model_info.get('protocol', case['protocol']),
            'display_preprocessing': case['protocol'],
            'canonical_shape': list(case['shape']), 'spacing_mm': case['spacing'].tolist(),
            'original_shape': list(case['original_shape']), 'original_affine': case['original_affine'].tolist(),
            'normalization': case['normalization'], 'measurements': measurements(pred, case)}


def overlay_png(case, pred):

    z = int(np.argmax((pred > 0).sum(axis=(0, 1)))) if np.any(pred) else pred.shape[2] // 2
    a = case['images'][z, 0].astype(float)
    a = np.clip((a + 5) / 10, 0, 1)
    rgb = np.repeat(a[..., None], 3, axis=2)
    labels = pred[:, :, z].T
    for label, color in [(1, [1, .59, .24]), (2, [.12, .71, .69]), (3, [.71, .48, .94])]:
        selected = labels == label
        rgb[selected] = .45 * rgb[selected] + .55 * np.array(color)
    out = io.BytesIO(); Image.fromarray(np.uint8(np.flipud(rgb) * 255)).save(out, format='PNG')
    return out.getvalue(), z


def analysis_bundle(case, pred, model_info, entropy=None, mi=None):
    
    payload = report_payload(case, pred, model_info)
    rows = payload['measurements']
    csv_text = io.StringIO(); writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    png, z = overlay_png(case, pred)
    import base64
    image = base64.b64encode(png).decode('ascii')
    headers = ''.join('<th>' + html.escape(k) + '</th>' for k in rows[0])
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row.values()) + '</tr>' for row in rows)
    document = f'''<!doctype html><meta charset="utf-8"><title>BrainTumor AI research report</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#163047}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:8px;text-align:left}}img{{max-width:480px;width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media print{{button{{display:none}}}}</style>
<h1>BrainTumor AI · Case report</h1><p>{html.escape(payload['created_utc'])}</p>
<h2>{payload['result']}</h2><p>{payload['interpretation']}</p>
<p>Research use only. Dice, when a reference is supplied, measures overlap; it is not classification accuracy.</p>
<table><tr>{headers}</tr>{table}</table>
<h2>FLAIR with predicted regions</h2><p>Canonical axial slice {z}. Orange: edema; teal: non-enhancing tumor; purple: enhancing tumor.</p><img src="data:image/png;base64,{image}">
<h2>Provenance</h2><pre>{html.escape(json.dumps(payload, indent=2))}</pre>
<p>Use your browser Print → Save as PDF to save a PDF copy.</p>'''
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.html', document)
        archive.writestr('analysis.json', json.dumps(payload, indent=2))
        archive.writestr('measurements.csv', csv_text.getvalue())
        archive.writestr('overlay.png', png)
        archive.writestr('segmentation_native.nii.gz', mask_bytes(pred, case))
        for name, array in [('entropy', entropy), ('mutual_information', mi)]:
            if array is not None:
                nii = nib.Nifti1Image(np.asarray(array, dtype=np.float32), case['affine'])
                nii.set_qform(case['affine'], 1); nii.set_sform(case['affine'], 1); nii.header.set_xyzt_units('mm')
                archive.writestr(name + '_canonical.nii.gz', gzip.compress(nii.to_bytes()))
    return buffer.getvalue()

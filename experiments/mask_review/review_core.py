import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_case(record, selected):
    source = Path(selected['path'])
    mask_path = Path(record['mask_path'])
    if sha(source) != record['input_sha256'] or sha(mask_path) != record['mask_sha256']:
        raise ValueError('Input or mask changed since extraction. Review the extraction report first.')
    image = nib.as_closest_canonical(nib.load(source))
    mask_image = nib.load(mask_path)
    if len(image.shape) != 3 or image.shape != mask_image.shape:
        raise ValueError('Expected matching 3D MRI and mask.')
    if not np.allclose(image.affine, mask_image.affine, atol=1e-4, rtol=0):
        raise ValueError('MRI and mask geometry differ; no automatic alignment performed.')
    a = image.get_fdata(dtype=np.float32)
    m = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(a).all() or not np.isfinite(m).all() or not np.isin(m, [0, 1]).all():
        raise ValueError('Invalid image or nonbinary mask.')
    values = a[a != 0]
    low, high = np.percentile(values, [1, 99]) if values.size else (0, 1)
    gray = (np.clip((a-low)/max(float(high-low), 1e-6), 0, 1)*255).astype('uint8')
    return gray, m.astype(bool), nib.affines.voxel_sizes(image.affine)


def panels(gray, mask, spacing, axis, index, opacity):
    g = np.flipud(np.take(gray, index, axis=axis).T)
    m = np.flipud(np.take(mask, index, axis=axis).T)
    rgb = np.repeat(g[..., None], 3, axis=2)
    overlay = rgb.astype(float)
    overlay[m] = (1-opacity)*overlay[m] + opacity*np.array([30, 220, 100])
    border = m & ~binary_erosion(m)
    overlay[border] = [255, 210, 30]
    extracted = rgb.copy(); extracted[~m] = 0
    remaining = [i for i in range(3) if i != axis]
    width_mm = g.shape[1]*spacing[remaining[0]]
    height_mm = g.shape[0]*spacing[remaining[1]]
    size = (420, max(1, round(420*height_mm/width_mm)))
    return [Image.fromarray(x.astype('uint8')).resize(size, Image.Resampling.NEAREST)
            for x in (rgb, overlay, extracted)]


def save_decision(path, case_id, decision):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[case_id] = decision
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)
    return data

import base64
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from imaging import normalize_display


@lru_cache(maxsize=1)
def plotly_js():
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


def mask_payload(mask, shape, affine):
    
    if mask is None:
        return '', [], []
    from skimage.measure import marching_cubes
    mask = np.asarray(mask)
    if mask.shape != shape or not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 2, 3]).all():
        raise ValueError('Viewer mask must match canonical MRI shape and use MSD labels 0,1,2,3.')
    colors = ['#ffb454', '#62ddca', '#dd85ff']
    names = ['Edema', 'Non-enhancing tumor', 'Enhancing tumor']
    meshes, volumes = [], []
    voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000
    for label, name, color in zip([1, 2, 3], names, colors):
        binary = mask == label
        count = int(binary.sum())
        volumes.append({'name': name, 'label': label, 'voxels': count, 'ml': count * voxel_ml})
        if not count:
            continue

        padded = np.pad(binary, [(0, n % 2) for n in shape])
        sx, sy, sz = padded.shape
        small = padded.reshape(sx//2, 2, sy//2, 2, sz//2, 2).any(axis=(1, 3, 5))
        vertices, faces, _, _ = marching_cubes(np.pad(small.astype(np.float32), 1), .5)
        vertices = (vertices - 1) * 2 + .5
        world = vertices @ affine[:3, :3].T + affine[:3, 3]
        meshes.append(dict(type='mesh3d', x=world[:,0].round(4).tolist(), y=world[:,1].round(4).tolist(),
                           z=world[:,2].round(4).tolist(), i=faces[:,0].tolist(), j=faces[:,1].tolist(),
                           k=faces[:,2].tolist(), color=color, opacity=.65, name=name, label=label,
                           flatshading=False, hoverinfo='name', lighting={'ambient':.55,'diffuse':.75}))
    return base64.b64encode(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).decode('ascii'), meshes, volumes


def build_viewer(case, modality, prediction=None):
    image = np.asarray(case['images'][:, modality].transpose(2,1,0))
    if image.ndim != 3 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError('Viewer requires a finite 3D MRI volume with at least 2 voxels per dimension.')
    affine = np.asarray(case['affine'], dtype=float)
    if affine.shape != (4,4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) < 1e-9:
        raise ValueError('Invalid MRI affine; cannot place surfaces in physical space.')
    encoded = np.rint(normalize_display(image,1,99)*255).astype(np.uint8)
    pred, pm, pv = mask_payload(prediction, image.shape, affine)
    ref, rm, rv = mask_payload(case.get('reference'), image.shape, affine)
    payload = dict(shape=list(image.shape), spacing=np.asarray(case['spacing']).tolist(),
                   affine=affine.tolist(), image=base64.b64encode(encoded.tobytes()).decode('ascii'),
                   mask=pred, reference=ref, labels=[1,2,3], meshes={'pred':pm,'ref':rm},
                   volumes={'pred':pv,'ref':rv})
    template = Path(__file__).with_name('viewer.html').read_text(encoding='utf-8')

    return template.replace('__VOLUME_PAYLOAD__', json.dumps(payload, allow_nan=False)).replace('__PLOTLY_JS__', plotly_js())

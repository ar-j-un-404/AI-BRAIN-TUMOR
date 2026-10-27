from pathlib import Path
import hashlib
import nibabel as nib
import numpy as np

MAX_VOXELS = 30_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_image(image):
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError('Only non-degenerate 3D volumes are supported; select one examination, not a 4D series.')
    if int(np.prod(image.shape, dtype=np.int64)) > MAX_VOXELS:
        raise ValueError('Volume exceeds the Stage 1 limit of 30 million voxels.')
    affine = image.affine
    if affine is None or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError('Missing, non-finite or singular spatial affine.')
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous affine.')
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode and not scode:
        raise ValueError('Neither qform nor sform has a spatial code. Resolve image geometry first.')
    if qcode and scode and not np.allclose(qform, sform, atol=1e-3, rtol=0):
        raise ValueError('qform and sform disagree. Resolve the header before displaying overlays.')
    if np.iscomplexobj(np.empty(0, dtype=image.get_data_dtype())):
        raise ValueError('Complex-valued images are unsupported.')


def load_volume(path, is_mask=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'File not found: {path}')
    if not (path.name.lower().endswith('.nii') or path.name.lower().endswith('.nii.gz')):
        raise ValueError('Choose a .nii or .nii.gz file.')
    image = nib.load(str(path))
    validate_image(image)
    original = {'shape': list(image.shape), 'orientation': list(nib.aff2axcodes(image.affine)),
                'affine': image.affine.tolist(), 'spacing': nib.affines.voxel_sizes(image.affine).tolist()}

    image = nib.as_closest_canonical(image)
    affine = image.affine
    spacing = nib.affines.voxel_sizes(affine)
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError('Oblique or sheared volume: Stage 1 requires axis-aligned images. No automatic resampling was performed.')
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('Image contains NaN or infinite values.')
    labels = []
    if is_mask:
        if np.any(data < 0) or not np.allclose(data, np.rint(data), atol=1e-5, rtol=0):
            raise ValueError('Annotation must contain nonnegative integer labels, not probabilities.')
        labels = np.unique(data).astype(int).tolist()
        if len(labels) > 64:
            raise ValueError('More than 64 annotation labels: this may be an intensity image.')
    unit = image.header.get_xyzt_units()[0]
    warnings = []
    if unit != 'mm':
        warnings.append(f'Spatial units are {unit!r}; measurements are not reported in millimetres.')
    if not is_mask and np.ptp(data) == 0:
        warnings.append('Constant-valued image: no intensity contrast.')
    return {'data': data, 'affine': affine, 'spacing': spacing, 'unit': unit,
            'metadata': {'filename': path.name, 'original': original,
                         'canonical_shape': list(data.shape), 'canonical_orientation': ['R', 'A', 'S'],
                         'canonical_affine': affine.tolist(), 'spacing': spacing.tolist(),
                         'spatial_unit': unit, 'labels': labels, 'warnings': warnings}}


def check_alignment(reference, other):
    if reference['data'].shape != other['data'].shape:
        raise ValueError('Shape mismatch after canonical reorientation. Overlay blocked.')
    if reference['unit'] != other['unit']:
        raise ValueError('Spatial-unit mismatch. Overlay blocked.')
    if not np.allclose(reference['affine'], other['affine'], atol=1e-3, rtol=0):
        raise ValueError('Spatial affine mismatch. Overlay blocked; registration/resampling must be reviewed separately.')


def normalize_display(data, low=1, high=99):
    
    if not 0 <= low < high <= 100:
        raise ValueError('Contrast percentiles must satisfy 0 <= low < high <= 100.')
    foreground = data[data != 0]
    values = foreground if foreground.size else data.ravel()
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0, 1)


def slice_plane(data, plane, index):
    axis = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}[plane]
    if not 0 <= index < data.shape[axis]:
        raise ValueError('Slice index is outside the volume.')
    return np.take(data, index, axis=axis).T


def label_volumes(mask):
    if mask['unit'] != 'mm':
        return None
    voxel_ml = abs(np.linalg.det(mask['affine'][:3, :3])) / 1000
    return {str(label): float(np.count_nonzero(mask['data'] == label) * voxel_ml)
            for label in mask['metadata']['labels'] if label != 0}

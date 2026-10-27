import csv
import hashlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

PROTOCOL = 'ras-axial-zcyx-f32-nonzero-zscore-clip5-v1'
MODALITIES = {'0': 'FLAIR', '1': 'T1w', '2': 't1gd', '3': 'T2w'}
LABELS = {'0': 'background', '1': 'edema', '2': 'non-enhancing tumor', '3': 'enhancing tumour'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes its root.')
    return path


def load_splits(project):
    split_dir = project / 'data/splits'
    manifest = read_json(split_dir / 'split_manifest.json')
    root = Path(manifest['dataset_root'])
    metadata_path = root / 'dataset.json'
    if sha256(metadata_path) != manifest['dataset_json_sha256']:
        raise ValueError('dataset.json changed after splitting.')
    metadata = read_json(metadata_path)
    if metadata['modality'] != MODALITIES or metadata['labels'] != LABELS:
        raise ValueError('This stage requires the reviewed MSD modality/label mapping.')
    partitions = {name: read_csv(split_dir / f'{name}.csv')
                  for name in ['train', 'validation', 'internal_test']}
    all_rows = [r for rows in partitions.values() for r in rows]
    all_ids = [r['case_id'] for r in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Repeated case IDs in frozen splits.')
    expected = {Path(r['image']).name.removesuffix('.nii.gz'): r for r in metadata['training']}
    if set(all_ids) != set(expected) or len(all_ids) != metadata['numTraining']:
        raise ValueError('Splits do not cover exactly the declared training cases.')
    for name, rows in partitions.items():
        if len(rows) != manifest['counts'][name]:
            raise ValueError(f'Partition count changed: {name}')
        for row in rows:
            cid = row['case_id']
            if not cid.replace('_', '').isalnum():
                raise ValueError('Unsafe case identifier.')
            if row['image'] != expected[cid]['image'] or row['mask'] != expected[cid]['label']:
                raise ValueError(f'Split paths changed: {cid}')
            if not row.get('image_voxel_sha256'):
                raise ValueError(f'Missing audit fingerprint: {cid}')
    fingerprints = [r['image_voxel_sha256'] for r in all_rows]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError('Duplicate audited image arrays across splits; review before training.')

    split_hashes = {n: sha256(split_dir / f'{n}.csv') for n in partitions}
    return root, partitions, split_hashes


def check_header(image):
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-8:
        raise ValueError('Invalid image affine.')
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if not qc and not sc:
        raise ValueError('Missing spatial header codes.')
    if qc and sc and not np.allclose(q, s, atol=1e-3, rtol=0):
        raise ValueError('qform/sform conflict.')
    if np.issubdtype(image.get_data_dtype(), np.complexfloating):
        raise ValueError('Complex-valued image.')


def prepare_case(root, row, cache_root):
    cid = row['case_id']
    folder = cache_root / cid
    image_path = source_path(root, row['image'])
    mask_path = source_path(root, row['mask'])
    source_hashes = {'image_file_sha256': sha256(image_path), 'mask_file_sha256': sha256(mask_path)}
    if (folder / 'metadata.json').is_file():
        saved = read_json(folder / 'metadata.json')
        if saved['protocol'] != PROTOCOL or saved['source_hashes'] != source_hashes:
            raise ValueError(f'Cache/source changed for {cid}. Preserve the old run and use a new cache directory.')
        if saved['audited_image_voxel_sha256'] != row['image_voxel_sha256']:
            raise ValueError(f'Audit fingerprint changed: {cid}')
        for name, digest in saved['cache_hashes'].items():
            if sha256(folder / name) != digest:
                raise ValueError(f'Cache corruption: {folder / name}')
        return saved
    if folder.exists():
        raise ValueError(f'Incomplete cache without metadata: {folder}')
    image, mask = nib.load(image_path), nib.load(mask_path)
    check_header(image); check_header(mask)
    if len(image.shape) != 4 or image.shape[3] != 4 or mask.shape != image.shape[:3]:
        raise ValueError(f'Invalid dimensions: {cid}')
    if not np.allclose(image.affine, mask.affine, atol=1e-3, rtol=0):
        raise ValueError(f'MRI/mask grid mismatch: {cid}')
    if image.header.get_xyzt_units()[0] != 'mm' or mask.header.get_xyzt_units()[0] != 'mm':
        raise ValueError(f'Millimetre units required: {cid}')
    digest = hashlib.sha256(str(image.shape).encode())
    for c in range(4):
        a = np.asarray(image.dataobj[..., c], dtype=np.float32)
        if not np.isfinite(a).all():
            raise ValueError(f'Non-finite intensity: {cid}')
        digest.update(np.ascontiguousarray(a, dtype='<f4').tobytes())
    del a
    if digest.hexdigest() != row['image_voxel_sha256']:
        raise ValueError(f'Image changed since audit: {cid}')
    image, mask = nib.as_closest_canonical(image), nib.as_closest_canonical(mask)
    spacing = nib.affines.voxel_sizes(image.affine)
    if not np.allclose(image.affine[:3,:3] / spacing, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError(f'Oblique/sheared image requires review: {cid}')
    if mask.shape != image.shape[:3] or not np.allclose(image.affine, mask.affine, atol=1e-3, rtol=0):
        raise ValueError('Canonical grid mismatch.')
    labels = mask.get_fdata(dtype=np.float32)
    if not np.isfinite(labels).all() or not np.allclose(labels, np.rint(labels), atol=1e-5, rtol=0):
        raise ValueError('Invalid mask values.')
    if not set(np.unique(labels)).issubset({0,1,2,3}) or not np.any(labels > 0):
        raise ValueError('Unexpected or empty mask; review required.')
    nx, ny, nz, _ = image.shape
    needed = nx*ny*nz*17 + 32*1024*1024
    cache_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache_root).free < needed:
        raise ValueError('Insufficient disk space for the next cached case.')
    tmp = cache_root / (cid + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    cached = np.lib.format.open_memmap(tmp/'images.npy', mode='w+', dtype=np.float32, shape=(nz,4,ny,nx))
    normalization = []
    for c in range(4):
        a = np.asarray(image.dataobj[...,c], dtype=np.float32)
        fg = a != 0
        if not fg.any():
            raise ValueError('Empty MRI channel.')
        mean, std = float(a[fg].mean(dtype=np.float64)), float(a[fg].std(dtype=np.float64))
        if std < 1e-8:
            raise ValueError('Constant MRI channel.')
        normalized = np.zeros_like(a)
        normalized[fg] = (a[fg]-mean)/std
        np.clip(normalized, -5, 5, out=normalized)
        cached[:,c] = normalized.transpose(2,1,0)
        normalization.append({'mean': mean, 'std': std})
    cached.flush(); del cached
    np.save(tmp/'masks.npy', labels.astype(np.uint8).transpose(2,1,0))
    info = {'case_id':cid, 'protocol':PROTOCOL, 'source_hashes':source_hashes,
            'audited_image_voxel_sha256':row['image_voxel_sha256'],
            'canonical_affine':image.affine.tolist(), 'canonical_shape':list(image.shape),
            'slice_count':nz, 'normalization':normalization, 'spacing':spacing.tolist(),
            'cache_hashes':{name:sha256(tmp/name) for name in ['images.npy','masks.npy']}}
    write_json(tmp/'metadata.json', info)
    tmp.rename(folder)
    return info


class SliceDataset(Dataset):
    
    def __init__(self, folders):
        self.folders = [Path(f) for f in folders]
        self.index = []
        self.open_arrays = OrderedDict()
        for i, f in enumerate(self.folders):
            n = read_json(f/'metadata.json')['slice_count']
            self.index.extend((i,z) for z in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i,z = self.index[index]
        if i not in self.open_arrays:
            f = self.folders[i]
            self.open_arrays[i] = (np.load(f/'images.npy', mmap_mode='r'),np.load(f/'masks.npy',mmap_mode='r'))
            if len(self.open_arrays)>4:
                self.open_arrays.popitem(last=False)
        self.open_arrays.move_to_end(i)
        x,y = self.open_arrays[i]
        return (torch.from_numpy(np.array(x[z], dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(y[z], dtype=np.int64, copy=True)))

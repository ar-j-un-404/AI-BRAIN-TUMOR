import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

REGIONS = {'whole_tumor': (1, 2, 3), 'tumor_core': (2, 3), 'enhancing_tumor': (3,)}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(path)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def csv_out(path, data):
    if not data: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(path, data):
    path = Path(path); tmp = path.with_suffix('.tmp'); torch.save(data, tmp); tmp.replace(path)

def code_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def load_checkpoint(path):
    if not Path(path).is_file(): raise FileNotFoundError(f'Checkpoint missing: {path}')
    return torch.load(path, map_location='cpu', weights_only=True)

import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parent

def fetch(url,path,algorithm,expected):
    if path.is_file() and hashlib.new(algorithm,path.read_bytes()).hexdigest()==expected:return
    tmp=path.with_suffix('.partial');digest=hashlib.new(algorithm);total=0
    with urllib.request.urlopen(url,timeout=90) as response,tmp.open('wb') as f:
        while True:
            block=response.read(1024*1024)
            if not block:break
            f.write(block);digest.update(block);total+=len(block)
            if total%(10*1024*1024)==0:print(f'Downloaded {total/1024/1024:.0f} MiB',flush=True)
    if digest.hexdigest()!=expected:raise ValueError('Official download checksum mismatch: '+path.name)
    tmp.replace(path)

def setup():
    assets=BASE/'assets';assets.mkdir(exist_ok=True)
    code=assets/'hd_bet-2.0.1.tar.gz'
    print('Fetching HD-BET 2.0.1 from PyPI...',flush=True)
    with urllib.request.urlopen('https://pypi.org/pypi/HD-BET/2.0.1/json',timeout=60) as r:info=json.load(r)
    source=next(x for x in info['urls'] if x['packagetype']=='sdist')
    fetch(source['url'],code,'sha256','46e3ad50b0c8d45b2b91cf2d2b821b3d26fd885ffa31628ad7687b5651a4838b')
    vendor=BASE/'vendor';vendor.mkdir(exist_ok=True)
    with tarfile.open(code) as t:
        for member in t.getmembers():
            parts=Path(member.name).parts[1:]
            if not member.isfile() or not parts:continue
            if parts[0]!='HD_BET' and not parts[0].lower().startswith('license'):continue
            out=vendor.joinpath(*parts)
            if not out.resolve().is_relative_to(vendor.resolve()):raise ValueError('Unsafe archive path')
            out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(t.extractfile(member).read())
    archive=assets/'release_v1.5.0.zip'
    print('Checking official HD-BET weights (114.7 MB first download)...',flush=True)
    fetch('https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1',archive,'md5','f97a36f89f8eb3071ff05c6f9ac66b56')
    root=assets/'model';root.mkdir(exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            out=root/member.filename
            if not out.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe model archive path')
            if member.is_dir():out.mkdir(parents=True,exist_ok=True)
            else:
                out.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as f,out.open('wb') as dest:
                    while block:=f.read(1024*1024):dest.write(block)
    candidates=[p.parent for p in root.rglob('plans.json') if (p.parent/'dataset.json').is_file() and (p.parent/'fold_all').is_dir()]
    if len(candidates)!=1:raise ValueError('Unexpected model layout: '+str(candidates))
    location=candidates[0]
    manifest=dict(model_folder=str(location.resolve()),software='HD-BET 2.0.1',
        source_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),model_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        model_archive_md5='f97a36f89f8eb3071ff05c6f9ac66b56',
        files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()},
        code_license='Apache-2.0; official source license included in vendor',
        weights_source='https://zenodo.org/records/14445620',weights_license_note='Repository software license is not asserted to separately license the weights; see official record. No weights redistributed in this pack.')
    (assets/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Official assets verified. Existing Python environments were not changed.',flush=True)

if __name__=='__main__':setup()

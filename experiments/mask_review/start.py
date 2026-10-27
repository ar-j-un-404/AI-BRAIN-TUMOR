import json
import subprocess
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
python = (Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2/.venv/Scripts/python.exe')
if not python.is_file():
    python = Path(input('Paste the main project .venv\\Scripts\\python.exe path: ').strip().strip('"'))
if not python.is_file():
    raise SystemExit('Python interpreter not found.')
check = subprocess.run([str(python), '-c', 'import streamlit,nibabel,numpy,scipy,PIL'])
if check.returncode:
    raise SystemExit('Missing dependencies. Install streamlit nibabel numpy scipy pillow in this environment.')

candidates = []
for root in Path.home().joinpath('Downloads').glob('BrainTumor_Brain_Extraction_QC*'):
    candidates.extend(root.glob('reports/*/cases.json'))
    candidates.extend(root.glob('*/reports/*/cases.json'))
if candidates and not (BASE/'settings.json').exists():
    latest = max(candidates, key=lambda p:p.stat().st_mtime_ns)
    (BASE/'settings.json').write_text(json.dumps({'report':str(latest.parent)}))
print('Opening review viewer: http://localhost:8509', flush=True)
sys.exit(subprocess.call([str(python), '-m', 'streamlit', 'run', str(BASE/'app.py'), '--server.port','8509', '--server.address','localhost'], cwd=BASE))

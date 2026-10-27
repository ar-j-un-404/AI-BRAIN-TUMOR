import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'app/settings.json'


def configure(project, classifier=None):
    project = Path(project).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f'Project folder does not exist: {project}')
    if not (project / 'research_suite').is_dir():
        raise ValueError('Select the inner project folder containing research_suite and runs.')
    if classifier and not (Path(classifier) / 'experiment.json').is_file():
        raise FileNotFoundError('Classifier results must contain experiment.json.')
    settings = {'project': str(project), 'classifier_results': str(Path(classifier).resolve()) if classifier else ''}
    SETTINGS.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print('Configuration saved locally. No model or MRI was copied.')
    return settings


def read_settings():
    if not SETTINGS.is_file():
        raise FileNotFoundError('Run python launch.py configure --project "YOUR_PROJECT_FOLDER" first.')
    return json.loads(SETTINGS.read_text(encoding='utf-8'))


def app_python(settings):
    project = Path(settings['project'])
    for path in (project / '.venv/Scripts/python.exe', project / '.venv/bin/python'):
        if path.is_file():
            return path
    return Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description='Configure and launch the unified app using an existing trained project.')
    sub = parser.add_subparsers(dest='command', required=True)
    config = sub.add_parser('configure')
    config.add_argument('--project', required=True)
    config.add_argument('--classifier', help='Original BTC_IXI_Training/results folder')
    sub.add_parser('check')
    start = sub.add_parser('start')
    start.add_argument('--port', type=int, default=8507)
    args = parser.parse_args()
    if args.command == 'configure':
        configure(args.project, args.classifier)
        return 0
    settings = read_settings()
    python = app_python(settings)
    if args.command == 'check':
        probe = 'import torch,nibabel,numpy,scipy,streamlit,plotly,skimage; print("App imports passed; CUDA:",torch.cuda.is_available())'
        return subprocess.call([str(python), '-c', probe])
    return subprocess.call([str(python), '-m', 'streamlit', 'run', str(ROOT/'app/app.py'), '--server.address', 'localhost', '--server.port', str(args.port)], cwd=ROOT)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

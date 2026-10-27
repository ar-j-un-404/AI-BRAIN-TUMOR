import ast
import hashlib
import io
import tokenize
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', ROOT/'launch.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

class RepositoryTests(unittest.TestCase):
    def test_local_configuration_and_missing_project(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            with patch.object(launcher, 'SETTINGS', p/'settings.json'):
                with self.assertRaises(ValueError): launcher.configure(p)
                (p/'research_suite').mkdir()
                config = launcher.configure(p)
                self.assertEqual(launcher.read_settings(), config)
                self.assertEqual(config['classifier_results'], '')
                with self.assertRaises(FileNotFoundError): launcher.configure(p, p/'missing')

    def test_classifier_source_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT/'app/classifier_core.py').read_bytes()).hexdigest(), '9db3da39c37c58f07b9c58b5ec376a642772f37a04189a04095fbb3d08f204e0')

    def test_comment_free_python(self):
        for folder in ('app', 'research_suite', 'experiments', 'tests'):
            for path in (ROOT/folder).rglob('*.py'):
                tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
                self.assertFalse(any(t.type == tokenize.COMMENT for t in tokens), str(path))

    def test_no_docstrings(self):
        paths = list(ROOT.glob('*.py'))
        for folder in ('app', 'research_suite', 'experiments', 'tests'):
            paths.extend((ROOT/folder).rglob('*.py'))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(ast.get_docstring(node), str(path))

    def test_original_classifier_hash_compatibility(self):
        spec = importlib.util.spec_from_file_location('contracts', ROOT/'app/contracts.py')
        contracts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contracts)
        exp = {'version': 'btc-ixi-paired-v1', 'preprocessing': contracts.PREPROCESSING,
               'steps': 1000, 'threshold': .5,
               'source_hashes': {'core.py': 'bc90080de5cbbde1b8be60acf11e229b662daaafe8213f5f4293092ed9adf7f0'},
               'records': [{'id': str(i), 'fold': i%5+1} for i in range(500)]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'experiment.json').write_text(json.dumps(exp))
            self.assertEqual(contracts.validate_experiment(root), exp)
            with patch.object(contracts, 'sha', return_value='modified-source'):
                with self.assertRaises(ValueError): contracts.validate_experiment(root)

    def test_results_and_page_routes(self):
        results = json.loads((ROOT/'app/results_summary.json').read_text())
        self.assertEqual(len(results['classification']), 4)
        self.assertEqual(results['classification'][0]['BTC balanced accuracy'], 59.09)
        self.assertEqual(results['segmentation'][0]['WT Dice'], 90.83)
        app = (ROOT/'app/app.py').read_text()
        self.assertIn("elif page=='Models & results':", app)
        self.assertIn("'Experiment comparisons'", app)
        self.assertNotIn("st.sidebar.caption('Project:", app)
        self.assertNotIn('No training runs or reserved test batches are started by this app.', app)

    def test_source_release_has_no_patient_assets(self):
        blocked = {'.pt', '.pth', '.pkl', '.npy', '.npz', '.nii', '.dcm', '.xls', '.xlsx', '.csv', '.tsv'}
        for p in ROOT.rglob('*'):
            if p.is_file() and '.git' not in p.parts:
                self.assertNotIn(p.suffix.lower(), blocked, str(p))
                self.assertFalse(p.name.endswith('.nii.gz'), str(p))

if __name__ == '__main__': unittest.main()

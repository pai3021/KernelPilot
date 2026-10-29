"""Check the isolated worker's import closure, without the control repo on sys.path."""
import os,shutil,subprocess,sys,tempfile,unittest,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class WorkerPackageTest(unittest.TestCase):
    def workspace(self,base,include_callback):
        (base/'scripts').mkdir();
        # Python3.10 tomli is an installed declared dependency, not control-repo code.
        if sys.version_info < (3,11):
            shutil.copytree(Path(importlib.util.find_spec('tomli').origin).parent,base/'tomli')
        shutil.copytree(ROOT/'benchmark_backend',base/'benchmark_backend',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT/'scripts/run_fixed_benchmark.py',base/'scripts/run_remote.py')
        shutil.copy2(ROOT/'experiments/harness-value/bundle_budget.py',base/'bundle_budget.py')
        if include_callback:shutil.copy2(ROOT/'scripts/prepare_codex_baseline.py',base/'scripts/prepare_codex_baseline.py')
    def check(self,include_callback):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.workspace(base,include_callback)
            return subprocess.run([sys.executable,'-I','scripts/run_remote.py','--help'],cwd=base,env={'PATH':os.environ.get('PATH','')},capture_output=True,text=True,timeout=10)
    def test_declared_callback_imports_in_isolated_worker(self):
        result=self.check(True);self.assertEqual(result.returncode,0,result.stderr)
    def test_missing_callback_detected_before_models_or_gpu(self):
        result=self.check(False);self.assertNotEqual(result.returncode,0);self.assertIn('scripts.prepare_codex_baseline',result.stderr)
if __name__=='__main__':unittest.main()

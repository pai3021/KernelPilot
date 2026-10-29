import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from benchmark_backend.base import BenchmarkExecutionResult
from scripts import run_fixed_benchmark as runner
class Backend:
    calls=[]
    def __init__(self,config):pass
    def evaluate_blob(self,blob,tasks,params,**kwargs):
        self.calls.append((json.loads(blob),tasks,params))
        leaf={'status':'PASSED','latency_ms':1.0,'reference_latency_ms':2.0,'speedup_factor':2.0,'max_abs_error':.00001,'max_rel_error':.00002,'log':''}
        return BenchmarkExecutionResult.from_results({tasks[0]:{tasks[0]:leaf}})
class FixedRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'solution').mkdir();(self.root/'.ako').mkdir()
        (self.root/'solution/kernel.py').write_text('class ModelNew:\n    pass\n')
        (self.root/'solution/remote-config.toml').write_text('[benchmark.remote]\nhost="test"\n')
        (self.root/'.ako/fixed-evaluator.json').write_text(json.dumps({'task':'level1/demo','params':{'correct_trials':3,'perf_trials':20,'validation_profile':'distribution-v2','seed':1729}}))
        (self.root/'baseline.json').write_text(json.dumps({'reference_latency_ms':99.0}))
        Backend.calls=[]
    def tearDown(self):self.tmp.cleanup()
    def invoke(self,args):
        reserve=Mock()
        env={'BUNDLE_COMPARISON_LEDGER':str(self.root/'global.json'),'BUNDLE_SCOPE_PREFIX':'task','BUNDLE_BACKEND_EVENTS':str(self.root/'events.jsonl')}
        with patch.object(runner,'ROOT',self.root),patch.dict(os.environ,env),patch.dict(sys.modules,{'bundle_budget':types.SimpleNamespace(reserve=reserve)}),patch('benchmark_backend.RemoteSSHBenchmarkBackend',Backend),patch.object(sys,'argv',['run_remote.py',*args]):
            runner.main()
        return reserve
    def test_one_call_fresh_reference_and_immutable_candidate_binding(self):
        reserve=self.invoke(['--label','iter1','--capture-logs'])
        self.assertEqual(len(Backend.calls),1);self.assertEqual(Backend.calls[0][2]['validation_profile'],'distribution-v2')
        reserve.assert_called_once()
        reports=list((self.root/'trajectory').glob('*/results.json'));self.assertEqual(len(reports),1)
        report=json.loads(reports[0].read_text());leaf=report['results']['level1/demo']['level1/demo']
        self.assertEqual(leaf['reference_latency_ms'],2.0);self.assertEqual(leaf['speedup_factor'],2.0)
        self.assertNotIn('max_abs_error',leaf)
        witness=report['evaluated_candidate'];self.assertEqual(Path(witness['artifact_path']).read_text(),Backend.calls[0][0]['sources']['kernel.py'])
    def test_workload_filters_rejected_before_dispatch(self):
        with self.assertRaises(SystemExit):self.invoke(['--label','bad','--first','1'])
        self.assertEqual(Backend.calls,[])
if __name__=='__main__':unittest.main()

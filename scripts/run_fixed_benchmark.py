#!/usr/bin/env python3
"""Fresh-reference single-candidate runner for controlled Harness experiments.

A child copy is installed as scripts/run_remote.py; the existing parent file
bridge still owns remote execution. Unlike bench_utils it never caches a parent
latency or dispatches an extra parent candidate.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
    bridge='EXPERIMENTPILOT_BENCHMARK_BRIDGE_DIR'
    server='EXPERIMENTPILOT_BENCHMARK_BRIDGE_SERVER'
    if os.environ.get(bridge) and not os.environ.get(server):
        raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).with_name('benchmark_bridge_client.py')),*sys.argv[1:]]))
    from benchmark_backend import RemoteSSHBenchmarkBackend
    from benchmark_backend.budget import begin_benchmark_scope_from_environment
    from benchmark_backend.provenance import capture_evaluated_candidate
    from scripts.prepare_codex_baseline import fixed_backend_evaluator
    from bundle_budget import reserve
    try:import tomllib
    except ImportError:import tomli as tomllib
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--capture-logs',action='store_true');parser.add_argument('--quiet',action='store_true');args=parser.parse_args()
    begin_benchmark_scope_from_environment()
    reserve(Path(os.environ['BUNDLE_COMPARISON_LEDGER']),os.environ['BUNDLE_SCOPE_PREFIX']+'/H1/search')
    binding=json.loads((ROOT/'.ako/fixed-evaluator.json').read_text())
    values=tomllib.loads((ROOT/'solution/remote-config.toml').read_text())['benchmark']['remote']
    aliases={'host':'remote_host','python':'remote_python','kernelbench':'remote_kernelbench','runs_dir':'remote_runs_dir'}
    backend=RemoteSSHBenchmarkBackend({aliases.get(k,k):v for k,v in values.items()})
    original=backend.evaluate_blob
    def observed(blob,tasks,params,**kwargs):
        start=time.monotonic();result=original(blob,tasks,params,**kwargs)
        with Path(os.environ['BUNDLE_BACKEND_EVENTS']).open('a') as out:
            out.write(json.dumps({'label':'H1/search','wall_seconds':time.monotonic()-start,'blob_sha256':hashlib.sha256(blob.encode()).hexdigest(),'result':asdict(result)},default=str)+'\n')
        return result
    backend.evaluate_blob=observed
    provenance=capture_evaluated_candidate(ROOT)
    source=Path(provenance['artifact_path']).read_text()
    feedback=dict(fixed_backend_evaluator(backend,task_id=binding['task'],evaluation_params=binding['params'])(source))
    leaf=dict(feedback)
    folder=ROOT/'trajectory'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.label)
    folder.mkdir(parents=True,exist_ok=False)
    report={'timestamp':datetime.now().isoformat(),'label':args.label,'backend':'ssh','reference_policy':'fresh-per-candidate','results':{binding['task']:{binding['task']:leaf}},'evaluated_candidate':provenance}
    (folder/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(feedback),flush=True)
    print('Evaluated candidate SHA-256: '+provenance['candidate_sha256'],flush=True)
    if feedback['status']!='PASSED':raise SystemExit(1)

if __name__=='__main__':main()

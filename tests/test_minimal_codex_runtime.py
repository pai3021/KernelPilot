from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from scripts.prepare_codex_baseline import prepare_workspace
from scripts.run_minimal_codex import native_command, run_native


class MinimalCodexRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "native"
        source = "class ModelNew: pass\n"
        witness = {"source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
        prepare_workspace(self.workspace, {"task_id": "test", "task": witness, "parent_candidate": witness,
                                            "environment": {}, "evaluation_policy": {}})

    def tearDown(self):
        self.temp.cleanup()

    def executable(self, body):
        path = self.root / "fake-codex"
        path.write_text("#!/usr/bin/python3\n" + textwrap.dedent(body))
        path.chmod(0o755)
        return str(path)

    def test_native_request_uses_requested_model_and_never_reads_the_harness_task(self):
        command = native_command("codex", self.workspace, "gpt-5.6-sol", "high")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-sol")
        self.assertIn('model_reasoning_effort="high"', command)
        self.assertIn("PROBLEM.md", command[-1])
        self.assertNotIn("CODEX_TASK", " ".join(command))

    def test_native_cli_and_endpoint_run_end_to_end_without_real_model_or_gpu(self):
        executable = self.executable('''
            import json, subprocess, sys
            from pathlib import Path
            assert '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1] == 'gpt-5.6-sol'
            assert not Path('CODEX_TASK.md').exists()
            assert not Path('skills').exists()
            print(json.dumps({'type':'thread.started','thread_id':'fake-native'}), flush=True)
            Path('solution/kernel.py').write_text('# chosen candidate\\n')
            subprocess.run([sys.executable, 'evaluate.py'], check=True)
            Path('.native-last-message.txt').write_text('done')
            print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':3}}), flush=True)
        ''')
        sources = []
        def evaluate(source):
            sources.append(source)
            return {"status": "PASSED", "correct": True, "latency_ms": 1.0}
        result = run_native(self.workspace, self.root / "observer", executable=executable,
                            model="gpt-5.6-sol", reasoning_effort="high", evaluate=evaluate, timeout=5)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["session_id"], "fake-native")
        self.assertEqual(result["submission_attempts"], 1)
        self.assertEqual(sources, ["# chosen candidate\n"])
        self.assertEqual(result["usage"], {"input_tokens": 12, "output_tokens": 3})

    def test_infrastructure_block_terminates_native_worker_and_does_not_retry(self):
        executable = self.executable('''
            import subprocess, sys, time
            subprocess.run([sys.executable, 'evaluate.py'])
            time.sleep(30)
        ''')
        sources = []
        def evaluate(source):
            sources.append(source)
            return {"status": "INFRA_FAILURE", "correct": None, "error_log": "busy"}
        result = run_native(self.workspace, self.root / "observer", executable=executable,
                            model="gpt-5.6-sol", reasoning_effort="high", evaluate=evaluate, timeout=5)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(sources), 1)

    def test_time_limit_and_existing_run_are_not_silently_reset(self):
        executable = self.executable('''
            import time
            time.sleep(30)
        ''')
        kwargs = dict(executable=executable, model="gpt-5.6-sol", reasoning_effort="high",
                      evaluate=lambda source: {"status": "PASSED"}, timeout=0.2)
        result = run_native(self.workspace, self.root / "observer", **kwargs)
        self.assertEqual(result["status"], "timed_out")
        with self.assertRaises(FileExistsError):
            run_native(self.workspace, self.root / "observer", **kwargs)

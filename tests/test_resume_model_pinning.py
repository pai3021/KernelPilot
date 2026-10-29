import unittest
from pathlib import Path
from agent_runtime.codex import CodexRuntime
from agent_runtime.types import AgentRunRequest
class ResumeModelPinning(unittest.TestCase):
    def test_resume_does_not_revert_to_global_model(self):
        runtime=CodexRuntime(template_root=Path(__file__).resolve().parents[1],executable='codex')
        request=AgentRunRequest(Path('/tmp/branch'),'continue',60,model='gpt-5.6-sol',environment={'AGENT_RUNTIME_REASONING_EFFORT':'high'})
        command=runtime.build_command(request,session_id='thread-to-resume',resume=True)
        self.assertEqual(command[command.index('--model')+1],'gpt-5.6-sol')
        self.assertIn('model_reasoning_effort="high"',command)
        self.assertEqual(command[-2:],['thread-to-resume','continue'])

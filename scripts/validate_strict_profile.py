"""CPU-only semantic witness for the opt-in V2 checker on real PyTorch."""
import json
import torch
import kernelbench.eval as evaluator
from scripts.strict_validation import validation_context

def main():
    if torch.cuda.is_available():raise RuntimeError('Run this CPU witness with CUDA_VISIBLE_DEVICES empty')
    class Reference(torch.nn.Module):
        def forward(self,a,b):return a@b
    class Additive(Reference):
        def forward(self,a,b):return super().forward(a,b)+.718
    class Multiplicative(Reference):
        def forward(self,a,b):return super().forward(a,b)*1.000702
    results={}
    with validation_context('distribution-v2',evaluator,torch):
        for name,model in [('reference',Reference()),('additive',Additive()),('multiplicative',Multiplicative())]:
            result=evaluator.run_and_check_correctness(Reference(),model,lambda:[torch.rand(32,32),torch.rand(32,32)],metadata={},num_correct_trials=3,seed=739,device=torch.device('cpu'),backend='triton',precision=torch.float32)
            results[name]={'correct':result.correctness,'max_abs_error':result.metadata.get('measured_max_abs_error')}
        label=torch.tensor([0,1],dtype=torch.int64)
        assert evaluator._process_input_tensor(label,torch.device('cpu')).dtype==torch.int64
    assert results['reference']['correct'] and not results['additive']['correct'] and not results['multiplicative']['correct']
    print(json.dumps({'status':'CPU_PROFILE_OK','gpu_available':False,'results':results}))
if __name__=='__main__':main()

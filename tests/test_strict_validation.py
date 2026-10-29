import contextlib
import types
import unittest
import numpy as np
from scripts.strict_validation import case_schedule, transform_inputs, validation_context

class Tensor(np.ndarray):
    def __new__(cls, value, dtype=np.float32):
        return np.asarray(value, dtype=dtype).view(cls)
    def is_floating_point(self): return np.issubdtype(self.dtype, np.floating)
    def to(self, **kwargs): return self
    def mul_(self, value): self *= value; return self
    def sub_(self, value): self -= value; return self
    def zero_(self): self.fill(0); return self
    def fill_(self, value): self.fill(value); return self
    def abs_(self): np.abs(self, out=self); return self
    def abs(self): return np.abs(self)
    def clamp_min_(self, value): np.maximum(self, value, out=self); return self
    def div_(self, value): np.divide(self, value, out=self); return self

class Model:
    def __init__(self, function=lambda x:x.copy()): self.function=function
    def to(self, **kwargs): return self
    def __call__(self, *args): return self.function(*args)

class StrictValidationTest(unittest.TestCase):
    def setUp(self):
        self.torch=types.SimpleNamespace(Tensor=Tensor, float32=np.float32,
            randint=lambda lo,hi,shape:Tensor(np.random.randint(lo,hi,shape),dtype=np.int64),
            no_grad=contextlib.nullcontext, allclose=np.allclose)
        self.original_checker=object()
        self.evaluator=types.SimpleNamespace(run_and_check_correctness=self.original_checker,
            _process_input_tensor=lambda value,*args:value,
            set_seed=np.random.seed,get_tolerance_for_precision=lambda p:1e-4,
            KernelExecResult=lambda **kw:types.SimpleNamespace(**kw))
    def run_case(self, function):
        with validation_context('distribution-v2',self.evaluator,self.torch):
            return self.evaluator.run_and_check_correctness(Model(),Model(function),
                lambda:[Tensor(np.random.rand(64))],metadata={},num_correct_trials=3,
                seed=517,device='cpu',precision=np.float32)
    def test_original_and_stress_cases_are_not_a_single_replayed_input(self):
        self.assertEqual(case_schedule(3),('original','original','original','signed','zeros','ones'))
        for bad in (0,-1,True,1.5):
            with self.assertRaises(ValueError):case_schedule(bad)
    def test_correct_candidate_records_measured_zero_error(self):
        result=self.run_case(lambda x:x.copy())
        self.assertTrue(result.correctness)
        self.assertEqual(result.metadata['measured_max_abs_error'],0.0)
        self.assertEqual(result.metadata['correctness_trials'],'(6 / 6)')
    def test_positive_only_implementation_is_rejected_on_signed_inputs(self):
        result=self.run_case(lambda x:np.maximum(x,0))
        self.assertFalse(result.correctness)
        self.assertGreater(result.metadata['measured_max_abs_error'],0.1)
    def test_zero_input_regression_is_rejected(self):
        result=self.run_case(lambda x:x+np.float32(.01) if np.all(x==0) else x.copy())
        self.assertFalse(result.correctness)
    def test_one_input_regression_is_rejected(self):
        result=self.run_case(lambda x:x*np.float32(1.000702) if np.all(x==1) else x.copy())
        self.assertFalse(result.correctness)
    def test_integer_labels_and_scalar_parameters_are_preserved(self):
        label=Tensor([1,2],dtype=np.int64)
        values=transform_inputs([Tensor([.25,.75]),label,7],'signed')
        np.testing.assert_array_equal(values[0],[-.5,.5])
        np.testing.assert_array_equal(label,[1,2]);self.assertEqual(values[2],7)
        with validation_context('distribution-v2',self.evaluator,self.torch):
            self.assertIs(self.evaluator._process_input_tensor(label,'cpu'),label)
    def test_callback_restored_after_failure(self):
        old_transfer=self.evaluator._process_input_tensor
        with self.assertRaises(RuntimeError):
            with validation_context('distribution-v2',self.evaluator,self.torch):raise RuntimeError('failure')
        self.assertIs(self.evaluator.run_and_check_correctness,self.original_checker)
        self.assertIs(self.evaluator._process_input_tensor,old_transfer)
    def test_unknown_profile_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            with validation_context('bad-profile',self.evaluator,self.torch):pass

if __name__=='__main__':unittest.main()

"""Versioned correctness checks for the multi-task V2 experiment.

The legacy KernelBench evaluator remains the timing/compilation owner. Only its
correctness callback and integer input transfer are scoped to this opt-in
profile. This is additional validation, not a hostile-code security boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

PROFILE = "distribution-v2"


def case_schedule(trials: int):
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("correctness trials must be positive")
    return ("original",) * trials + ("signed", "zeros", "ones")


def transform_inputs(inputs, family):
    if family not in {"original", "signed", "zeros", "ones"}:
        raise ValueError("Unknown validation family")
    for value in inputs:
        if not hasattr(value, "is_floating_point") or not value.is_floating_point():
            continue
        if family == "signed":
            value.mul_(2).sub_(1)
        elif family == "zeros":
            value.zero_()
        elif family == "ones":
            value.fill_(1)
    return inputs


def make_checker(evaluator, torch):
    def checker(original_model_instance, new_model_instance, get_inputs_fn,
                metadata, num_correct_trials=3, verbose=False, seed=42,
                device=None, backend="triton", precision=None):
        precision = torch.float32 if precision is None else precision
        tolerance = evaluator.get_tolerance_for_precision(precision)
        cases = case_schedule(num_correct_trials)
        evaluator.set_seed(seed)
        seeds = [int(torch.randint(0, 2**31 - 1, (1,)).item()) for _ in cases]
        maximum_abs = maximum_rel = 0.0
        metadata.update(validation_profile=PROFILE, validation_checks=len(cases),
                        relative_error_denominator_floor=1e-8)
        is_cuda = str(device).startswith("cuda") or isinstance(device, int)
        with torch.no_grad():
            for index, (family, trial_seed) in enumerate(zip(cases, seeds)):
                try:
                    evaluator.set_seed(trial_seed)
                    inputs = [evaluator._process_input_tensor(x, device, backend, precision)
                              for x in get_inputs_fn()]
                    transform_inputs(inputs, family)
                    evaluator.set_seed(trial_seed)
                    model = original_model_instance.to(device=device, dtype=precision)
                    evaluator.set_seed(trial_seed)
                    candidate = new_model_instance.to(device=device, dtype=precision)
                    expected = model(*inputs)
                    if is_cuda:
                        torch.cuda.synchronize(device=device)
                    actual = candidate(*inputs)
                    if is_cuda:
                        torch.cuda.synchronize(device=device)
                    if (not isinstance(expected, torch.Tensor)
                            or not isinstance(actual, torch.Tensor)
                            or expected.shape != actual.shape or expected.dtype != actual.dtype):
                        metadata["correctness_issue"] = "Output shape/type mismatch"
                        return evaluator.KernelExecResult(compiled=True, correctness=False,
                                                          metadata=metadata)
                    passed = bool(torch.allclose(expected, actual, atol=tolerance, rtol=tolerance))
                    delta = (expected - actual).abs_()
                    absolute = float(delta.max().item())
                    relative = float(delta.div_(expected.abs().clamp_min_(1e-8)).max().item())
                    maximum_abs = max(maximum_abs, absolute) if math.isfinite(absolute) else None
                    maximum_rel = max(maximum_rel, relative) if math.isfinite(relative) else None
                    metadata.update(measured_max_abs_error=maximum_abs,
                                    measured_max_rel_error=maximum_rel,
                                    correctness_trials=f"({index + int(passed)} / {len(cases)})")
                    if not passed:
                        metadata["correctness_issue"] = "Fixed validation protocol mismatch"
                        return evaluator.KernelExecResult(compiled=True, correctness=False,
                                                          metadata=metadata)
                    # Do not retain GB-sized inputs and outputs across trials.
                    del delta, expected, actual, inputs
                except Exception as exc:
                    metadata.update(runtime_error=str(exc), runtime_error_name=type(exc).__name__)
                    return evaluator.KernelExecResult(compiled=True, correctness=False,
                                                      metadata=metadata)
        return evaluator.KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return checker


@contextmanager
def validation_context(profile, evaluator, torch):
    if profile is None:
        yield
        return
    if profile != PROFILE:
        raise ValueError("Unknown validation profile")
    old_checker = evaluator.run_and_check_correctness
    old_transfer = evaluator._process_input_tensor
    def transfer(value, device, backend="cuda", precision=None):
        if isinstance(value, torch.Tensor) and not value.is_floating_point():
            return value.to(device=device)
        return old_transfer(value, device, backend, precision or torch.float32)
    evaluator.run_and_check_correctness = make_checker(evaluator, torch)
    evaluator._process_input_tensor = transfer
    try:
        yield
    finally:
        evaluator.run_and_check_correctness = old_checker
        evaluator._process_input_tensor = old_transfer

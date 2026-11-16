"""Fast Walsh-Hadamard Incoherence Rotation (`QuaRot` / `SpinQuant`) and OCP `FP8` codecs."""

from quant_forge.transforms.hadamard_fp8 import (
    fast_walsh_hadamard,
    quantize_fp8_e4m3,
    quantize_fp8_e5m2,
    run_hadamard_and_fp8_benchmark,
)

__all__ = [
    "fast_walsh_hadamard",
    "quantize_fp8_e4m3",
    "quantize_fp8_e5m2",
    "run_hadamard_and_fp8_benchmark",
]

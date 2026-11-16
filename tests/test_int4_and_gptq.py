"""Unit and Hypothesis property tests for C++20 INT4 bit-packing, NumPy fallback, GPTQ, and AWQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_forge.kernels.native_int4 import (
    QuantKernelBridge,
    benchmark_packed_int4_kernel,
    get_quant_bridge,
)
from quant_forge.quantizers.gptq_awq import (
    quantize_awq_int4,
    quantize_gptq_int4,
    quantize_rtn_int4,
    run_gptq_vs_awq_benchmark,
)


def test_cpp20_vs_numpy_int4_packing_and_fused_gemm_parity() -> None:
    cpp_b = get_quant_bridge()
    py_b = QuantKernelBridge(csrc_path=Path("/tmp/missing_quant_forge_kernel.cpp"))
    assert cpp_b.is_native_cpp is True
    assert "c++20-native" in cpp_b.backend
    assert py_b.is_native_cpp is False
    assert py_b.backend == "numpy-fallback"

    with pytest.raises(ValueError, match="group_size"):
        cpp_b.pack_int4_grouped(np.zeros((8, 32), dtype=np.float32), group_size=3)

    rng = np.random.default_rng(11)
    w = rng.normal(0.0, 0.2, size=(32, 64)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(10, 64)).astype(np.float32)

    pk_cpp = cpp_b.pack_int4_grouped(w, group_size=16)
    pk_py = py_b.pack_int4_grouped(w, group_size=16)

    np.testing.assert_array_equal(pk_cpp.packed_u8, pk_py.packed_u8)
    np.testing.assert_allclose(pk_cpp.scales, pk_py.scales, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pk_cpp.zeros, pk_py.zeros, rtol=1e-5, atol=1e-6)

    y_cpp = cpp_b.fused_int4_gemm(x, pk_cpp)
    y_py = py_b.fused_int4_gemm(x, pk_py)
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="Input dim"):
        cpp_b.fused_int4_gemm(np.zeros((4, 16), dtype=np.float32), pk_cpp)

    bench = benchmark_packed_int4_kernel(out_features=64, in_features=128, group_size=32)
    assert bench.compression_ratio_x > 5.0
    assert bench.cpp_vs_dequant_max_abs_diff < 1e-5


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=10000),
    group_size=st.sampled_from([8, 16, 32]),
)
def test_hypothesis_fused_cpp20_gemm_matches_explicit_dequant_matrix_multiply(
    seed: int, group_size: int
) -> None:
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.5, 1.5, size=(16, 32)).astype(np.float32)
    x = rng.uniform(-2.0, 2.0, size=(6, 32)).astype(np.float32)
    bridge = get_quant_bridge()
    packed = bridge.pack_int4_grouped(w, group_size=group_size)
    y_fused = bridge.fused_int4_gemm(x, packed)
    y_explicit = x @ packed.dequantize_f32().T
    np.testing.assert_allclose(y_fused, y_explicit, rtol=1e-4, atol=1e-4)


def test_gptq_and_awq_beat_rtn_under_heavy_outliers() -> None:
    with pytest.raises(ValueError, match="divisible"):
        quantize_rtn_int4(torch.randn(8, 30), group_size=16)
    with pytest.raises(ValueError, match="divisible"):
        quantize_gptq_int4(torch.randn(8, 30), torch.randn(16, 30), group_size=16)

    # Also exercise act_order=False branch in GPTQ
    x_cal = torch.randn(64, 32)
    w_sm = torch.randn(16, 32) * 0.1
    w_no_perm = quantize_gptq_int4(w_sm, x_cal, group_size=16, act_order=False)
    assert w_no_perm.shape == (16, 32)

    w_awq, s_vec, gamma = quantize_awq_int4(w_sm, x_cal, group_size=16, grid_points=11)
    assert w_awq.shape == (16, 32)
    assert s_vec.shape == (32,)
    assert 0.0 <= gamma <= 1.0

    rep = run_gptq_vs_awq_benchmark(out_features=128, in_features=128, group_size=32, seed=29)
    assert rep.awq_metrics.output_mse < rep.rtn_metrics.output_mse
    assert rep.gptq_metrics.output_mse < rep.awq_metrics.output_mse
    assert rep.gptq_metrics.relative_error_reduction_vs_rtn_x > 20.0

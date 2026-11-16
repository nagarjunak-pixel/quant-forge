"""Orthogonal Walsh-Hadamard Incoherence Rotation (`QuaRot` Ashkboos et al. 2024 /
`SpinQuant` Liu et al. 2024) & Bit-Exact OCP `FP8` (`E4M3` vs `E5M2`) Microscaling Codecs.
"""

from __future__ import annotations

import math

import torch

from quant_forge.quantizers.gptq_awq import quantize_rtn_int4
from quant_forge.types import HadamardFP8Report


def fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """In-register O(D log D) orthonormal Sylvester Fast Walsh-Hadamard Transform along `dim=-1`.

    Satisfies `FWHT(FWHT(x)) == x` and `<FWHT(u), FWHT(v)> == <u, v>` for power-of-two `D`.
    """
    dim = x.shape[-1]
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Last dimension ({dim}) must be a positive power of 2")

    h = x.clone()
    step = 1
    while step < dim:
        h = h.reshape(*x.shape[:-1], dim // (2 * step), 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.stack((a + b, a - b), dim=-2)
        step *= 2
    out: torch.Tensor = h.reshape_as(x) / math.sqrt(float(dim))
    return out


def _quantize_ieee_fp8_generic(
    x: torch.Tensor,
    *,
    exp_bits: int,
    mant_bits: int,
    max_normal: float,
) -> tuple[torch.Tensor, float]:
    """Exact bit-level OCP FP8 (`E4M3` max=448.0 or `E5M2` max=57344.0) normal+subnormal simulator."""
    bias = (1 << (exp_bits - 1)) - 1
    min_normal_exp = 1 - bias
    subnormal_step = math.ldexp(1.0, min_normal_exp - mant_bits)

    abs_x = x.abs()
    clipped_mask = abs_x > max_normal
    clipped_pct = float(clipped_mask.float().mean().item()) * 100.0

    clamped = torch.clamp(abs_x, max=max_normal)
    sign = torch.sign(x)

    # Subnormal region: |x| < 2^(1 - bias)
    min_normal_val = math.ldexp(1.0, min_normal_exp)
    is_sub = clamped < min_normal_val

    sub_q = torch.round(clamped / subnormal_step) * subnormal_step

    safe_norm = torch.where(is_sub, torch.full_like(clamped, min_normal_val), clamped)
    exp_unbiased = torch.floor(torch.log2(safe_norm))
    mant_scale = torch.pow(2.0, exp_unbiased - float(mant_bits))
    norm_q = torch.clamp(torch.round(safe_norm / mant_scale) * mant_scale, max=max_normal)

    out = sign * torch.where(is_sub, sub_q, norm_q)
    return out, round(clipped_pct, 4)


def quantize_fp8_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """OCP `FP8 E4M3` (1 sign, 4 exponent, 3 mantissa bits, bias=7, max finite=448.0)."""
    return _quantize_ieee_fp8_generic(x, exp_bits=4, mant_bits=3, max_normal=448.0)


def quantize_fp8_e5m2(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """OCP `FP8 E5M2` (1 sign, 5 exponent, 2 mantissa bits, bias=15, max finite=57344.0)."""
    return _quantize_ieee_fp8_generic(x, exp_bits=5, mant_bits=2, max_normal=57344.0)


def _crest_factor(t: torch.Tensor) -> float:
    rms = float(t.pow(2).mean().sqrt().item())
    peak = float(t.abs().max().item())
    return peak / max(rms, 1e-12)


def run_hadamard_and_fp8_benchmark(
    *,
    dim: int = 128,
    batch_size: int = 128,
    seed: int = 37,
) -> HadamardFP8Report:
    """Prove QuaRot/SpinQuant Hadamard rotation invariance `(XH)(WH)^T == XW^T`, measure outlier
    crest-factor suppression & INT4 MSE reduction, and compare `FP8 E4M3` vs `E5M2` dynamic range.
    """
    torch.manual_seed(seed)
    x = torch.randn(batch_size, dim) * 0.6
    # Inject massive sparse outlier spikes into both activations and weights
    x[:, 7] *= 28.0
    x[:, 63] *= -24.0

    w = torch.randn(dim, dim) * 0.08
    w[:, 7] *= 9.5
    w[:, 63] *= -8.5

    # 1. Exact orthogonal rotational invariance before quantization
    x_rot = fast_walsh_hadamard(x)
    w_rot = fast_walsh_hadamard(w)

    y_exact = x @ w.T
    y_rot_exact = x_rot @ w_rot.T
    inv_diff = float((y_exact - y_rot_exact).abs().max().item())

    pre_crest = _crest_factor(x)
    post_crest = _crest_factor(x_rot)

    # 2. INT4 quantization MSE before vs after Walsh-Hadamard incoherence rotation
    w_q_raw = quantize_rtn_int4(w, group_size=32)
    w_q_rot = quantize_rtn_int4(w_rot, group_size=32)

    mse_unrot = float((y_exact - (x @ w_q_raw.T)).pow(2).mean().item())
    mse_rot = float((y_exact - (x_rot @ w_q_rot.T)).pow(2).mean().item())

    # 3. FP8 E4M3 vs E5M2 on wide-dynamic-range backward activation gradients (with spikes up to 1,800)
    grad_tensor = torch.randn(batch_size, dim) * 35.0
    grad_tensor[:6, :4] *= 32.0  # Peaks ~1,100..2,500 exceed E4M3 (448) but fit E5M2 (57,344)!

    q_e4m3, clip_e4 = quantize_fp8_e4m3(grad_tensor)
    q_e5m2, clip_e5 = quantize_fp8_e5m2(grad_tensor)

    g_pow = float(grad_tensor.pow(2).mean().item())
    err_e4 = max(1e-12, float((grad_tensor - q_e4m3).pow(2).mean().item()))
    err_e5 = max(1e-12, float((grad_tensor - q_e5m2).pow(2).mean().item()))
    sqnr_e4 = 10.0 * math.log10(g_pow / err_e4)
    sqnr_e5 = 10.0 * math.log10(g_pow / err_e5)

    return HadamardFP8Report(
        dimension=dim,
        orthogonal_invariance_max_diff=round(inv_diff, 7),
        pre_rotation_crest_factor=round(pre_crest, 2),
        post_rotation_crest_factor=round(post_crest, 2),
        outlier_crest_reduction_x=round(pre_crest / max(post_crest, 1e-8), 2),
        unrotated_int4_mse=round(mse_unrot, 6),
        hadamard_rotated_int4_mse=round(mse_rot, 6),
        int4_mse_improvement_x=round(mse_unrot / max(mse_rot, 1e-12), 2),
        fp8_e4m3_sqnr_db=round(sqnr_e4, 2),
        fp8_e5m2_sqnr_db=round(sqnr_e5, 2),
        fp8_overflow_clipped_e4m3_pct=clip_e4,
        fp8_overflow_clipped_e5m2_pct=clip_e5,
    )

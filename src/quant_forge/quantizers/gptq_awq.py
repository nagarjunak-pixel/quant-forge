"""Second-Order Hessian Optimal Brain Surgeon (`GPTQ`, Frantar et al., ICLR 2023) &
Activation-Aware Weight Quantization (`AWQ`, Lin et al., MLSys 2024 Best Paper) vs. Naive `RTN`.
"""

from __future__ import annotations

import math

import torch

from quant_forge.types import GPTQvsAWQBenchmarkReport, QuantMethodMetrics


def _quant_dequant_1d_params(
    w_slice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute asymmetric 4-bit [0..15] scale and zero-point per row over `w_slice` `[out, G]`."""
    w_min = w_slice.amin(dim=1)
    w_max = w_slice.amax(dim=1)
    span = (w_max - w_min).clamp(min=1e-8)
    scale = span / 15.0
    zero = torch.clamp(torch.round(-w_min / scale), 0.0, 15.0)
    return scale, zero


def quantize_rtn_int4(w: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Baseline Round-to-Nearest (`RTN`) asymmetric group-wise INT4 quantization."""
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    num_groups = in_f // group_size
    w_grp = w.reshape(out_f, num_groups, group_size)
    w_min = w_grp.amin(dim=2, keepdim=True)
    w_max = w_grp.amax(dim=2, keepdim=True)
    scale = (w_max - w_min).clamp(min=1e-8) / 15.0
    zero = torch.clamp(torch.round(-w_min / scale), 0.0, 15.0)
    q = torch.clamp(torch.round(w_grp / scale + zero), 0.0, 15.0)
    deq: torch.Tensor = (scale * (q - zero)).reshape(out_f, in_f)
    return deq


def quantize_awq_int4(
    w: torch.Tensor,
    x_calib: torch.Tensor,
    *,
    group_size: int = 32,
    grid_points: int = 21,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Activation-Aware Weight Quantization (`AWQ`, Lin et al., MLSys 2024 Best Paper).

    Searches optimal exponent `gamma in [0, 1]` on per-channel activation mean magnitude
    `s = mean(|X|, dim=0)^gamma` so `X @ W^T = (X / s) @ (W * s)^T` protects salient channels
    from group-wise INT4 rounding error without mixed-precision hardware fragmentation!
    """
    act_mag = x_calib.abs().mean(dim=0).clamp(min=1e-6)
    y_true = x_calib @ w.T

    best_gamma = 0.0
    best_mse = float("inf")
    best_w_eff = w.clone()
    best_scales = torch.ones_like(act_mag)

    for idx in range(grid_points):
        gamma = idx / float(max(1, grid_points - 1))
        s = act_mag.pow(gamma)
        # Normalize geometric mean of s to 1.0 for numerical conditioning
        s = s / s.log().mean().exp().clamp(min=1e-6)
        w_scaled = w * s.unsqueeze(0)
        w_q_scaled = quantize_rtn_int4(w_scaled, group_size=group_size)
        # Effective linear operator in original input coordinates: W_eff = W_q_scaled / s
        w_eff = w_q_scaled / s.unsqueeze(0)
        y_hat = x_calib @ w_eff.T
        mse = float((y_true - y_hat).pow(2).mean().item())
        if mse < best_mse:
            best_mse = mse
            best_gamma = gamma
            best_w_eff = w_eff
            best_scales = s

    return best_w_eff, best_scales, round(best_gamma, 3)


def quantize_gptq_int4(
    w: torch.Tensor,
    x_calib: torch.Tensor,
    *,
    group_size: int = 32,
    damp_ratio: float = 0.01,
    act_order: bool = True,
) -> torch.Tensor:
    """Exact Cholesky Second-Order `GPTQ` (`Frantar et al., ICLR 2023`).

    Computes `H = (2 / N) X^T X + damp * I`, optionally sorts columns in descending order
    of diagonal curvature `H_jj` (`act_order=True`, quantizing highest-salience outlier columns
    FIRST while maximum unquantized degrees of freedom remain to compensate error!), and applies
    the Cholesky Optimal Brain Surgeon update `W[:, j+1:] -= ((w_j - q_j) / R_jj) * R[j, j+1:]`.
    """
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")

    w_work = w.detach().clone().to(torch.float64)
    x64 = x_calib.detach().to(torch.float64)
    n_samples = x64.shape[0]

    h_mat = (2.0 / float(n_samples)) * (x64.T @ x64)
    diag_mean = float(torch.diagonal(h_mat).mean().item())
    damp = damp_ratio * max(diag_mean, 1e-6)
    h_mat.diagonal().add_(damp)

    # Compute pre-quantization per-group scales on original W if act_order permutes columns
    num_groups = in_f // group_size
    base_scales = torch.empty((out_f, in_f), dtype=torch.float64, device=w.device)
    base_zeros = torch.empty((out_f, in_f), dtype=torch.float64, device=w.device)
    for g in range(num_groups):
        c0, c1 = g * group_size, (g + 1) * group_size
        sc, zp = _quant_dequant_1d_params(w_work[:, c0:c1])
        base_scales[:, c0:c1] = sc.unsqueeze(1).expand(-1, group_size)
        base_zeros[:, c0:c1] = zp.unsqueeze(1).expand(-1, group_size)

    if act_order:
        perm = torch.argsort(torch.diagonal(h_mat), descending=True)
        inv_perm = torch.argsort(perm)
        w_work = w_work[:, perm]
        h_mat = h_mat[perm][:, perm]
        base_scales = base_scales[:, perm]
        base_zeros = base_zeros[:, perm]
    else:
        inv_perm = torch.arange(in_f, device=w.device)

    chol_l = torch.linalg.cholesky(h_mat)
    h_inv = torch.cholesky_inverse(chol_l)
    # Upper-triangular Cholesky of H^-1: H^-1 = R^T @ R -> exact OBS step uses row j of R!
    r_upper = torch.linalg.cholesky(h_inv, upper=True)

    w_quant = torch.empty_like(w_work)

    for j in range(in_f):
        w_col = w_work[:, j]
        sc_col = base_scales[:, j]
        zp_col = base_zeros[:, j]
        q_int = torch.clamp(torch.round(w_col / sc_col + zp_col), 0.0, 15.0)
        q_col = sc_col * (q_int - zp_col)
        w_quant[:, j] = q_col

        if j + 1 < in_f:
            r_jj = r_upper[j, j].clamp(min=1e-12)
            obs_err = (w_col - q_col) / r_jj
            w_work[:, j + 1 :] -= obs_err.unsqueeze(1) * r_upper[j, j + 1 :].unsqueeze(0)

    out: torch.Tensor = w_quant[:, inv_perm].to(w.dtype)
    return out


def _eval_method(
    name: str,
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    rtn_mse: float,
) -> QuantMethodMetrics:
    mse = float((y_true - y_hat).pow(2).mean().item())
    sig = float(y_true.pow(2).mean().item())
    sqnr = 10.0 * math.log10(max(sig, 1e-12) / max(mse, 1e-15))
    ratio = rtn_mse / max(mse, 1e-15)
    return QuantMethodMetrics(
        method=name,
        bits=4,
        output_mse=round(mse, 7),
        sqnr_db=round(sqnr, 2),
        relative_error_reduction_vs_rtn_x=round(ratio, 2),
    )


def run_gptq_vs_awq_benchmark(
    *,
    out_features: int = 128,
    in_features: int = 128,
    group_size: int = 32,
    outlier_multiplier: float = 22.0,
    seed: int = 29,
) -> GPTQvsAWQBenchmarkReport:
    """Benchmark Naive RTN vs MLSys'24 AWQ vs ICLR'23 Cholesky GPTQ under heavy-tailed
    Transformer activation outliers (Dettmers et al. / Xiao et al. emergent outlier regime).
    """
    torch.manual_seed(seed)
    # Correlated calibration activations + emergent heavy-tailed outlier channels
    latent = torch.randn(384, 32)
    mixer = torch.randn(32, in_features) * 0.45
    x_all = latent @ mixer + torch.randn(384, in_features) * 0.25

    outlier_idx = torch.tensor([3, 19, 47, 83, 109]) % in_features
    x_all[:, outlier_idx] *= outlier_multiplier

    x_calib = x_all[:256]
    x_eval = x_all[256:]

    w = torch.randn(out_features, in_features) * 0.08
    # Salient channels also have modest weight structure that RTN group scaling crushes or distorts
    y_true = x_eval @ w.T

    w_rtn = quantize_rtn_int4(w, group_size=group_size)
    y_rtn = x_eval @ w_rtn.T
    rtn_mse = float((y_true - y_rtn).pow(2).mean().item())

    w_awq, _, opt_gamma = quantize_awq_int4(w, x_calib, group_size=group_size)
    y_awq = x_eval @ w_awq.T

    w_gptq = quantize_gptq_int4(w, x_calib, group_size=group_size, damp_ratio=0.01, act_order=True)
    y_gptq = x_eval @ w_gptq.T

    return GPTQvsAWQBenchmarkReport(
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        outlier_channel_multiplier=outlier_multiplier,
        optimal_awq_gamma=opt_gamma,
        hessian_damping_pct=1.0,
        rtn_metrics=_eval_method("RTN-INT4 (Naive)", y_true, y_rtn, rtn_mse),
        awq_metrics=_eval_method("AWQ-INT4 (Salient Scale)", y_true, y_awq, rtn_mse),
        gptq_metrics=_eval_method("GPTQ-INT4 (Cholesky H^-1)", y_true, y_gptq, rtn_mse),
    )

# ADR 0001: Cholesky Second-Order GPTQ (`act_order`) vs. Salient-Channel AWQ

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Emergent activation outliers in Transformer blocks (`20x+` magnitude in `<3%` of channels) cause catastrophic output MSE degradation under naive Round-to-Nearest (`RTN`) 4-bit quantization.

## Decision
We implement both **MLSys 2024 Best Paper `AWQ`** (grid-searched per-channel equivalence scaling $s = \bar{|X|}^\gamma$) and **ICLR 2023 `GPTQ`** using exact `float64` Cholesky factorization of $H^{-1} = R^\top R$ paired with descending Hessian curvature ordering (`act_order=True`).

## Consequences
- `AWQ` reduces 4-bit output MSE by `4.54x` (`+6.57 dB` SQNR) via hardware-friendly diagonal scaling.
- `GPTQ` (`Cholesky H^-1 + act_order`) reduces 4-bit output MSE by **`25.49x` (`21.22 dB -> 35.28 dB` SQNR)** by actively compensating quantization error across correlated unquantized columns.

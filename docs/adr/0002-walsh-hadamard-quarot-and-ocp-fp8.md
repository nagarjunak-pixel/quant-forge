# ADR 0002: Fast Walsh-Hadamard Incoherence Rotation & Hybrid OCP FP8 (`E4M3` / `E5M2`)

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Weight-only compensation cannot eliminate extreme activation spikes when quantizing both weights and activations (`W4A4` / `W8A8`), and uniform `FP8 E4M3` overflows (`max=448.0`) on heavy-tailed backward pass gradients.

## Decision
We implement an $O(d \log d)$ orthonormal Sylvester Fast Walsh-Hadamard Transform (`fast_walsh_hadamard`) following `QuaRot` / `SpinQuant` alongside bit-accurate OCP `FP8 E4M3` (forward) and `FP8 E5M2` (backward, `max=57344.0`) codecs.

## Consequences
- Suppresses activation crest factor by `7.49x` (`24.56 -> 3.28`), cutting `INT4` output MSE by `3.34x` while preserving exact bilinear invariance (`max |Δ| = 6.87e-5`).
- Demonstrates why hybrid FP8 training uses `E5M2` for backward gradients (`26.26 dB` SQNR, `0.00%` clipping vs `E4M3` `5.79 dB` due to saturation above `448.0`).

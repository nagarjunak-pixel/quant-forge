# QuantForge — Educational Reference Implementation of LLM Weight Quantization (GPTQ, AWQ, Hadamard Rotation, FP8, Packed INT4)

[![QuantForge Frontier CI](https://github.com/nagarjunak-pixel/quant-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/nagarjunak-pixel/quant-forge/actions/workflows/ci.yml)
[![Release v1.0.0](https://img.shields.io/badge/release-v1.0.0-1f6feb.svg)](https://github.com/nagarjunak-pixel/quant-forge/releases/tag/v1.0.0)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB.svg)](https://www.python.org/)
[![C++20 Packed INT4](https://img.shields.io/badge/C%2B%2B20--O3-2%20weights%2Fbyte-00599C.svg)](csrc/quant_kernels.cpp)
[![PyTorch Cholesky](https://img.shields.io/badge/PyTorch-Cholesky%20H%E2%81%BB%C2%B9-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-10b981.svg)](LICENSE)

**LLM weight-quantization reference implementation — GPTQ, AWQ, Walsh-Hadamard rotation, simulated FP8, and a small C++20 packed-INT4 GEMM. Educational.**

**QuantForge** is a small, clean-room reimplementation of published post-training quantization techniques — GPTQ (Frantar et al.), AWQ (Lin et al.), Walsh-Hadamard incoherence rotation as used in QuaRot/SpinQuant, and OCP FP8 `E4M3`/`E5M2` rounding — written in PyTorch/NumPy, plus a ~115-line C++20 kernel for packed INT4 storage and dequant-fused matmul. All experiments run on small synthetic tensors on CPU; it is meant for studying the algorithms, not as a replacement for production toolkits such as AutoGPTQ, AutoAWQ, or llama.cpp.

1. **Packed INT4 storage and fused dequant-GEMM in C++20 (`csrc/quant_kernels.cpp`)**: Packs pairs of asymmetric 4-bit weights into `uint8_t` bytes (`6.40x` byte reduction at `G=64` on a `256×512` matrix: `524,288 B -> 81,920 B`) and computes $\sum s_g(q_i - z_g)x_i = s_g(\sum q_i x_i - z_g \sum x_i)$ without materializing the `float32` weight matrix (`max |Δ| = 7.2e-7` vs explicit dequantization). Scalar C++ loops loaded via `ctypes`, with a NumPy fallback when no compiler is available.
2. **GPTQ (`act_order`) vs. AWQ vs. RTN (`src/quant_forge/quantizers/gptq_awq.py`)**: On a synthetic `128×128` layer with injected `22x` activation-outlier channels, AWQ's per-channel scale grid search ($\gamma^\star = 0.45$) gives `4.54x` lower output MSE than round-to-nearest, and GPTQ's Cholesky-based error compensation ($H^{-1} = R^\top R$, `act_order=True`) gives `25.49x` lower MSE (`21.22 dB -> 35.28 dB` SQNR, `+14.06 dB`).
3. **Walsh-Hadamard rotation and simulated FP8 (`E4M3` / `E5M2`)**: An $O(d \log d)$ fast Walsh-Hadamard transform satisfying $(X H_d)(W H_d)^\top = X W^\top$ (`|Δ| = 6.87e-5`) reduces the crest factor of a synthetic outlier-heavy activation matrix by `7.49x` (`24.56 -> 3.28`) and INT4 MSE by `3.34x`. A bit-level FP8 rounding simulator illustrates why `E5M2` (`26.26 dB` SQNR, `0.00%` overflow) suits heavy-tailed gradients better than `E4M3` (`5.79 dB`, saturates at `448.0`).
4. **Analytical roofline model and a small Ollama check**: A Williams-Waterman-Patterson roofline calculation from published H100 SXM5 specs estimates `3.55x` INT4 speedup for memory-bound decode and `2.00x` FP8 speedup for compute-bound prefill (analytical estimates, not GPU measurements). A separate 6-question check against a local Ollama `qwen3.5:4b` (`Q4_K_M`, quantized by Ollama/llama.cpp, not by this repo) records `3.16 GB` vs an estimated `8.60 GB` FP16 (`2.72x`) and `78.7 tok/s` decode.

---

## Live Interactive Quantization & Roofline Workbench

![QuantForge Live Workbench — C++20 Packed INT4, Cholesky GPTQ/AWQ, SpinQuant FWHT & H100 Roofline](docs/assets/workbench-preview.png)

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph Incoherence["Stage 1: Incoherence & Salient Scaling"]
        I1["O(D log D) Walsh-Hadamard\n(XH)(WH)ᵀ = XWᵀ (-7.49x Crest)"] --> I2["AWQ Channel Equivalence\ns = mean(|X|)^γ* (4.54x MSE Gain)"]
    end

    subgraph SecondOrder["Stage 2: Second-Order Hessian Calibration"]
        S1["Empirical H = (2/N)XᵀX + λI\nCurvature Permutation (act_order)"] --> S2["Upper Cholesky H⁻¹ = RᵀR\nOBS Column Update (25.49x MSE Gain)"]
    end

    subgraph Kernel["Stage 3: Native C++20 Sub-Byte Runtime"]
        K1["2 Weights/Byte uint8 Packing\n6.40x Byte Reduction (5.0 bpw)"] --> K2["Zero-Materialization Register GEMM\ns_g(Σ q_i x_i - z_g Σ x_i)"]
    end

    subgraph Roofline["Stage 4: Hardware Roofline & Live Audit"]
        R1["H100 / M3 Max / A100 Roofline\nDecode 3.55x | Prefill FP8 2.00x"]
        R2["Live Qwen-3.5-4B Q4_K_M\n3.16 GB (2.72x) @ 78.7 tok/s"]
    end

    Incoherence --> SecondOrder --> Kernel --> Roofline
```

---

## Measured Reproducible Benchmarks

| Subsystem | CLI Command | Measured Result | Verified Invariant |
| :--- | :--- | :--- | :--- |
| **Native `C++20` Packed `INT4` GEMM** (`256×512`, `G=64`) | `uv run quant benchmark-int4` | **`524,288 B -> 81,920 B` (`6.40x`)** | `5.00` eff bits/w (FP32 scale/zero); `\|Δ\| = 7.2e-7` vs explicit dequant |
| **Cholesky `GPTQ` vs. `AWQ` vs. `RTN`** (`22x` outliers) | `uv run quant benchmark-gptq` | **`GPTQ`: `25.49x` lower MSE** (`35.28 dB`) · **`AWQ`: `4.54x`** (`27.79 dB`) | Strictly monotonic MSE: `GPTQ (0.0297) < AWQ (0.1666) < RTN (0.7565)` |
| **Walsh-Hadamard `SpinQuant` & OCP `FP8`** (`D=128`) | `uv run quant benchmark-hadamard` | **Crest `24.56 -> 3.28` (`7.49x`)** · **`3.34x` INT4 MSE gain** | Exact FWHT isometry `<Hu,Hv> == <u,v>`; `E5M2` `26.26 dB` vs `E4M3` `5.79 dB` |
| **H100 SXM5 Hardware Roofline** (`Ridge = 295.22 F/B`) | `uv run quant roofline` | **Decode `INT4`: `3.55x`** · **Prefill `FP8`: `2.00x`** | Proves `W4A16` memory-bandwidth speedup vs `W8A8` compute-bound speedup |
| **Live `qwen3.5:4b` (`Q4_K_M`) Hardware Audit** | `uv run quant eval-ollama --live-only` | **`3.16 GB` (`2.72x` vs FP16) · `78.7 tok/s` · `4/6` (`66.7%`)** | Saved in [`benchmarks/real_qwen_quant.json`](benchmarks/real_qwen_quant.json) |

## Scope & Limitations

- **Small codebase:** ~1.5k lines of Python (`src/quant_forge`, 1,481 lines including CLI, FastAPI app, and types) plus a ~115-line C++20 helper (`csrc/quant_kernels.cpp`). The C++ kernel is plain scalar loops (no SIMD/CUDA) loaded via `ctypes`.
- **Synthetic, small benchmarks:** GPTQ/AWQ/RTN, Hadamard, and FP8 results come from seeded random tensors (e.g. a `128×128` layer, `D=128`) with hand-injected outlier channels — no real model checkpoints are quantized and no perplexity or downstream accuracy is measured.
- **No GPU measurements:** everything runs on CPU. The H100/A100/M3 Max roofline speedups are analytical estimates from published peak FLOP/s and bandwidth figures, not measured kernels.
- **Live LLM check is tiny and external:** the Ollama audit is 6 self-written numeric questions (`4/6` correct) against `qwen3.5:4b`, whose `Q4_K_M` quantization was produced by Ollama/llama.cpp rather than by QuantForge; its size and tok/s figures describe that model, not this code.
- **Simplified vs. the original methods:** single-layer, single-pass versions of GPTQ/AWQ (no blocked lazy updates, no full-model calibration pipeline, no learned SpinQuant rotations; FP8 is simulated in `float32`). Not production-hardened.

---

## Quickstart

```bash
uv sync --python 3.13 --all-extras

# Run Ruff, Mypy --strict, and Pytest + Hypothesis suite (98.42% coverage achieved)
uv run ruff check src tests
uv run mypy src tests
uv run pytest --cov=quant_forge --cov-report=term-missing --cov-fail-under=95

# Run CLI benchmarks
uv run quant demo
uv run quant eval-ollama

# Build & launch the QuantForge Workbench UI on http://127.0.0.1:8050
cd ui && npm install && npm run build && cd ..
uv run quant serve --host 127.0.0.1 --port 8050
```

## Documentation & Mathematical Proofs

- [Mathematical Foundations & Hardware Proofs](docs/mathematical-proofs.md)
- [ADR 0001: Cholesky Second-Order GPTQ (`act_order`) vs. Salient-Channel AWQ](docs/adr/0001-cholesky-gptq-and-salient-awq.md)
- [ADR 0002: Walsh-Hadamard Incoherence Rotation & Hybrid OCP FP8](docs/adr/0002-walsh-hadamard-quarot-and-ocp-fp8.md)
- [ADR 0003: Native C++20 Bit-Packed INT4 Register-Fused GEMM](docs/adr/0003-native-cpp20-packed-int4-fused-gemm.md)
- [Security Policy](SECURITY.md)

## License

Apache License 2.0 — Copyright (c) 2026 Nagarjuna Kamireddy.

---

## Author & Enterprise AI Architecture Hub
Built by **[Nagarjuna Reddy Kamireddy](https://nagarjunak-pixel.github.io/)** (*Nagarjuna Kamireddy / Nagarjuna K*) — Lead AI Architect, GenAI & Agentic AI Lead, and Corporate AI Trainer (20+ years across Enterprise Data, SAP HANA & Production AI; AWS Certified Solutions Architect – Professional & NVIDIA-Certified Professional: Agentic AI).

- **Official Portfolio & Architecture Hub:** [https://nagarjunak-pixel.github.io/](https://nagarjunak-pixel.github.io/)
- **Enterprise Case Studies & Technical Articles:** [Case Studies](https://nagarjunak-pixel.github.io/case-studies.html) · [Architecture Articles](https://nagarjunak-pixel.github.io/articles.html) · [10-Mandate Role Fit](https://nagarjunak-pixel.github.io/roles.html)
- **Master CV & AI Search Index:** [Master CV](https://nagarjunak-pixel.github.io/resume.html) · [`llms-full.txt`](https://nagarjunak-pixel.github.io/llms-full.txt) · [LinkedIn](https://www.linkedin.com/in/nagarjuna-kamireddy)

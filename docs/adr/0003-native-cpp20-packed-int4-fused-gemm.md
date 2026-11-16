# ADR 0003: Native C++20 Bit-Packed INT4 (`2 weights/byte`) Register-Fused GEMM

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Simulated "fake quantization" in PyTorch stores 4-bit weights inside `float32` tensors (`32 bits/weight`), saving zero physical RAM and hiding bit-packing/nibble-unpacking bugs.

## Decision
We implement physical 2-weights-per-byte `uint8_t` bit-packing and algebraically fused register Dequant-GEMM in `csrc/quant_kernels.cpp` (`clang++ -O3 -std=c++20`), cached by SHA-256 hash in `.quant_cache/` with an exact NumPy fallback.

## Consequences
- Achieves **`6.40x` physical byte reduction** (`524,288 B -> 81,920 B` at $G=64$, `5.00` effective bits/weight including FP32 group scales and zeros; `4.50` bits/weight with FP16 scales/zeros) with **`7.2e-7` max `|Δ|`** vs explicit float32 dequantization.

# Security Policy

## Memory & Input Safety Controls in `quant-forge`

1. **Bounds-Checked Native C++20 Buffers:**
   `QuantKernelBridge` validates contiguous memory layout (`np.ascontiguousarray`), positive even `group_size` divisibility (`in_features % group_size == 0`), and exact input dimension matching (`x.shape[1] == qweight.in_features`) in Python *before* crossing the C ABI boundary into `csrc/quant_kernels.cpp`.
2. **Typed FastAPI Request Validation:**
   All calibration and Roofline endpoints enforce strict Pydantic `ge`/`le` bounds on tensor dimensions, batch sizes, and accelerator identifiers.

## Reporting a Vulnerability
Please report security issues responsibly to `sendemailtonagarjuna@gmail.com`.

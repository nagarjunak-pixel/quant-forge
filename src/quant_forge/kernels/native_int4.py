"""SHA-256 cached C++20 compiler & ctypes bridge for packed INT4 (2 weights/byte) fused GEMM."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from quant_forge.types import PackedINT4BenchmarkReport

F32Array = npt.NDArray[np.float32]
U8Array = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class PackedINT4Weight:
    packed_u8: U8Array  # [out_features, in_features // 2]
    scales: F32Array  # [out_features, in_features // group_size]
    zeros: F32Array  # [out_features, in_features // group_size]
    out_features: int
    in_features: int
    group_size: int

    def total_storage_bytes(self) -> int:
        return int(self.packed_u8.nbytes + self.scales.nbytes + self.zeros.nbytes)

    def dequantize_f32(self) -> F32Array:
        """Materialize exact float32 reconstruction `scale * (q - zero)` for verification."""
        low = (self.packed_u8 & 0x0F).astype(np.float32)
        high = ((self.packed_u8 >> 4) & 0x0F).astype(np.float32)
        interleaved = np.empty((self.out_features, self.in_features), dtype=np.float32)
        interleaved[:, 0::2] = low
        interleaved[:, 1::2] = high

        num_groups = self.in_features // self.group_size
        q_grp = interleaved.reshape(self.out_features, num_groups, self.group_size)
        s_exp = self.scales[:, :, np.newaxis]
        z_exp = self.zeros[:, :, np.newaxis]
        w_hat = s_exp * (q_grp - z_exp)
        return w_hat.reshape(self.out_features, self.in_features).astype(np.float32)


class QuantKernelBridge:
    """Compiles `csrc/quant_kernels.cpp` (`-O3 -std=c++20`) with exact NumPy fallback."""

    def __init__(self, csrc_path: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._src = csrc_path or (repo_root / "csrc" / "quant_kernels.cpp")
        self._lib: ctypes.CDLL | None = None
        self._backend = "numpy-fallback"
        self._try_compile_and_load()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_native_cpp(self) -> bool:
        return self._lib is not None

    def _try_compile_and_load(self) -> None:
        if not self._src.exists():
            return
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            return

        cache_dir = self._src.parent.parent / ".quant_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_hash = hashlib.sha256(self._src.read_bytes()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        out_lib = cache_dir / f"libquant_kernels_{src_hash}.{ext}"

        if not out_lib.exists():
            cmd = [
                compiler,
                "-O3",
                "-std=c++20",
                "-shared",
                "-fPIC",
                str(self._src),
                "-o",
                str(out_lib),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return

        lib = ctypes.CDLL(str(out_lib))
        f_ptr = ctypes.POINTER(ctypes.c_float)
        u8_ptr = ctypes.POINTER(ctypes.c_uint8)

        lib.qforge_pack_int4_grouped_f32.argtypes = [
            f_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            u8_ptr,
            f_ptr,
            f_ptr,
        ]
        lib.qforge_pack_int4_grouped_f32.restype = None

        lib.qforge_int4_fused_gemm_f32.argtypes = [
            f_ptr,
            u8_ptr,
            f_ptr,
            f_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            f_ptr,
        ]
        lib.qforge_int4_fused_gemm_f32.restype = None

        self._lib = lib
        self._backend = f"c++20-native ({Path(compiler).name} -O3)"

    def pack_int4_grouped(self, weight_f32: F32Array, group_size: int = 32) -> PackedINT4Weight:
        w = np.ascontiguousarray(weight_f32, dtype=np.float32)
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        if group_size < 2 or group_size % 2 != 0 or in_f % group_size != 0:
            raise ValueError(
                f"group_size ({group_size}) must be positive even and divide in_features ({in_f})"
            )

        num_groups = in_f // group_size
        packed = np.empty((out_f, in_f // 2), dtype=np.uint8)
        scales = np.empty((out_f, num_groups), dtype=np.float32)
        zeros = np.empty((out_f, num_groups), dtype=np.float32)

        if self._lib is not None:
            self._lib.qforge_pack_int4_grouped_f32(
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_f,
                in_f,
                group_size,
                packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                zeros.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return PackedINT4Weight(packed, scales, zeros, out_f, in_f, group_size)

        w_grp = w.reshape(out_f, num_groups, group_size)
        w_min = np.min(w_grp, axis=2)
        w_max = np.max(w_grp, axis=2)
        span = np.maximum(w_max - w_min, np.float32(1e-8))
        scales[:] = (span / np.float32(15.0)).astype(np.float32)
        zeros[:] = np.clip(np.round(-w_min / scales), 0.0, 15.0).astype(np.float32)

        q = np.clip(
            np.round(w_grp / scales[:, :, np.newaxis] + zeros[:, :, np.newaxis]),
            0.0,
            15.0,
        ).astype(np.uint8)
        q_flat = q.reshape(out_f, in_f)
        packed[:] = (q_flat[:, 0::2] & 0x0F) | ((q_flat[:, 1::2] & 0x0F) << 4)
        return PackedINT4Weight(packed, scales, zeros, out_f, in_f, group_size)

    def fused_int4_gemm(self, x_f32: F32Array, qweight: PackedINT4Weight) -> F32Array:
        x = np.ascontiguousarray(x_f32, dtype=np.float32)
        bsz, in_f = int(x.shape[0]), int(x.shape[1])
        if in_f != qweight.in_features:
            raise ValueError(f"Input dim {in_f} != weight in_features {qweight.in_features}")

        y = np.empty((bsz, qweight.out_features), dtype=np.float32)
        if self._lib is not None:
            self._lib.qforge_int4_fused_gemm_f32(
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                qweight.packed_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                qweight.scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                qweight.zeros.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bsz,
                qweight.out_features,
                qweight.in_features,
                qweight.group_size,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return y

        w_deq = qweight.dequantize_f32()
        return np.asarray(x.astype(np.float64) @ w_deq.astype(np.float64).T, dtype=np.float32)


_GLOBAL_BRIDGE: QuantKernelBridge | None = None


def get_quant_bridge() -> QuantKernelBridge:
    global _GLOBAL_BRIDGE
    if _GLOBAL_BRIDGE is None:
        _GLOBAL_BRIDGE = QuantKernelBridge()
    return _GLOBAL_BRIDGE


def benchmark_packed_int4_kernel(
    *,
    out_features: int = 256,
    in_features: int = 512,
    group_size: int = 64,
    batch_size: int = 16,
    seed: int = 19,
) -> PackedINT4BenchmarkReport:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.08, size=(out_features, in_features)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(batch_size, in_features)).astype(np.float32)

    bridge = get_quant_bridge()
    packed = bridge.pack_int4_grouped(w, group_size=group_size)
    y_fused = bridge.fused_int4_gemm(x, packed)

    w_deq = packed.dequantize_f32()
    y_ref_deq = (x.astype(np.float64) @ w_deq.astype(np.float64).T).astype(np.float32)
    y_fp32_true = (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)

    max_diff = float(np.max(np.abs(y_fused - y_ref_deq)))
    sig_power = float(np.mean(y_fp32_true.astype(np.float64) ** 2))
    err_power = max(1e-15, float(np.mean((y_fp32_true - y_fused).astype(np.float64) ** 2)))
    sqnr_db = 10.0 * float(np.log10(sig_power / err_power))

    fp32_bytes = int(w.nbytes)
    total_packed = packed.total_storage_bytes()
    ratio = fp32_bytes / float(total_packed)
    eff_bits = (total_packed * 8.0) / float(out_features * in_features)

    return PackedINT4BenchmarkReport(
        kernel_backend=bridge.backend,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        fp32_bytes=fp32_bytes,
        packed_total_bytes=total_packed,
        compression_ratio_x=round(ratio, 2),
        effective_bits_per_weight=round(eff_bits, 2),
        cpp_vs_dequant_max_abs_diff=round(max_diff, 8),
        output_sqnr_db=round(sqnr_db, 2),
    )

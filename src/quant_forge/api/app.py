"""FastAPI control plane and Sub-Byte Quantization & Roofline Workbench backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quant_forge.kernels.native_int4 import benchmark_packed_int4_kernel, get_quant_bridge
from quant_forge.quantizers.gptq_awq import run_gptq_vs_awq_benchmark
from quant_forge.roofline.profiler import analyze_hardware_roofline, default_quant_artifact_path
from quant_forge.transforms.hadamard_fp8 import run_hadamard_and_fp8_benchmark
from quant_forge.types import (
    GPTQvsAWQBenchmarkReport,
    HadamardFP8Report,
    HardwareRooflineReport,
    PackedINT4BenchmarkReport,
    RealOllamaQuantReport,
)


class GPTQRequest(BaseModel):
    out_features: int = Field(default=128, ge=32, le=512)
    in_features: int = Field(default=128, ge=32, le=512)
    group_size: int = Field(default=32, ge=16, le=128)
    outlier_multiplier: float = Field(default=22.0, ge=2.0, le=60.0)
    seed: int = Field(default=29)


class HadamardRequest(BaseModel):
    dim: int = Field(default=128, ge=16, le=512)
    batch_size: int = Field(default=128, ge=16, le=512)
    seed: int = Field(default=37)


class RooflineRequest(BaseModel):
    accelerator: str = Field(default="h100-sxm5")
    d_model: int = Field(default=4096, ge=512, le=16384)
    d_ff: int = Field(default=11008, ge=1024, le=65536)


_WORKBENCH_CACHE: dict[str, Any] | None = None


def _load_ollama_snapshot() -> dict[str, Any] | None:
    artifact = default_quant_artifact_path()
    if not artifact.exists():
        return None
    rep = RealOllamaQuantReport.model_validate(json.loads(artifact.read_text(encoding="utf-8")))
    return rep.model_dump()


def _get_workbench_cache() -> dict[str, Any]:
    global _WORKBENCH_CACHE
    if _WORKBENCH_CACHE is None:
        bridge = get_quant_bridge()
        int4_rep = benchmark_packed_int4_kernel()
        gptq_rep = run_gptq_vs_awq_benchmark()
        had_rep = run_hadamard_and_fp8_benchmark()
        roof_rep = analyze_hardware_roofline("h100-sxm5")
        _WORKBENCH_CACHE = {
            "service": "quant-forge",
            "version": "1.0.0",
            "kernel_backend": bridge.backend,
            "is_native_cpp": bridge.is_native_cpp,
            "packed_int4": int4_rep.model_dump(),
            "gptq_vs_awq": gptq_rep.model_dump(),
            "hadamard_fp8": had_rep.model_dump(),
            "roofline": roof_rep.model_dump(),
            "real_qwen_quant": _load_ollama_snapshot(),
        }
    else:
        _WORKBENCH_CACHE["real_qwen_quant"] = _load_ollama_snapshot()
    return _WORKBENCH_CACHE


def create_app(ui_dist_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="QuantForge Sub-Byte Compression, Hessian GPTQ/AWQ & Native C++20 INT4 Engine",
        version="1.0.0",
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        bridge = get_quant_bridge()
        return {
            "status": "ok",
            "service": "quant-forge",
            "version": "1.0.0",
            "kernel_backend": bridge.backend,
            "is_native_cpp": bridge.is_native_cpp,
        }

    @app.get("/api/v1/overview")
    async def overview() -> dict[str, Any]:
        return _get_workbench_cache()

    @app.get("/api/v1/int4/benchmark", response_model=PackedINT4BenchmarkReport)
    async def int4_bench() -> PackedINT4BenchmarkReport:
        return benchmark_packed_int4_kernel()

    @app.post("/api/v1/gptq/benchmark", response_model=GPTQvsAWQBenchmarkReport)
    async def gptq_bench(req: GPTQRequest) -> GPTQvsAWQBenchmarkReport:
        return run_gptq_vs_awq_benchmark(
            out_features=req.out_features,
            in_features=req.in_features,
            group_size=req.group_size,
            outlier_multiplier=req.outlier_multiplier,
            seed=req.seed,
        )

    @app.post("/api/v1/hadamard/benchmark", response_model=HadamardFP8Report)
    async def hadamard_bench(req: HadamardRequest) -> HadamardFP8Report:
        return run_hadamard_and_fp8_benchmark(
            dim=req.dim,
            batch_size=req.batch_size,
            seed=req.seed,
        )

    @app.post("/api/v1/roofline/analyze", response_model=HardwareRooflineReport)
    async def roofline_analyze(req: RooflineRequest) -> HardwareRooflineReport:
        return analyze_hardware_roofline(
            accelerator=req.accelerator,
            d_model=req.d_model,
            d_ff=req.d_ff,
        )

    resolved_dist = ui_dist_dir or (Path(__file__).resolve().parents[3] / "ui" / "dist")
    assets_dir = resolved_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_model=None)
    async def root_page() -> FileResponse | JSONResponse:
        index_html = resolved_dist / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return JSONResponse(
            {
                "service": "quant-forge",
                "mode": "headless-api",
                "docs": "/docs",
                "overview": "/api/v1/overview",
            }
        )

    return app


app = create_app()

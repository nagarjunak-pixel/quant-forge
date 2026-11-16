"""Tests for Fast Walsh-Hadamard Incoherence, OCP FP8, Roofline, Ollama Audit, API, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from quant_forge.api.app import create_app
from quant_forge.cli import app as cli_app
from quant_forge.roofline.profiler import analyze_hardware_roofline, run_real_ollama_quant_audit
from quant_forge.transforms.hadamard_fp8 import (
    fast_walsh_hadamard,
    quantize_fp8_e4m3,
    quantize_fp8_e5m2,
    run_hadamard_and_fp8_benchmark,
)


@settings(max_examples=25, deadline=None)
@given(
    dim=st.sampled_from([8, 16, 32, 64]),
    seed=st.integers(min_value=1, max_value=9999),
)
def test_hypothesis_walsh_hadamard_involution_and_inner_product_preservation(
    dim: int, seed: int
) -> None:
    torch.manual_seed(seed)
    u = torch.randn(5, dim)
    v = torch.randn(5, dim)
    hu = fast_walsh_hadamard(u)
    hv = fast_walsh_hadamard(v)
    # Involution: H(H(u)) == u
    u_rec = fast_walsh_hadamard(hu)
    assert float((u - u_rec).abs().max().item()) < 1e-5
    # Isometry: <Hu, Hv> == <u, v>
    dots_orig = (u * v).sum(dim=-1)
    dots_rot = (hu * hv).sum(dim=-1)
    assert float((dots_orig - dots_rot).abs().max().item()) < 1e-4


def test_hadamard_validation_and_fp8_e4m3_vs_e5m2_subnormals_and_clipping() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        fast_walsh_hadamard(torch.randn(4, 12))

    # Include zeros, tiny subnormals, normal values, and huge >448 values
    vals = torch.tensor([0.0, 1e-4, -0.125, 448.0, 1200.0, -60000.0])
    q4, clip4 = quantize_fp8_e4m3(vals)
    q5, clip5 = quantize_fp8_e5m2(vals)
    assert float(q4.abs().max().item()) == 448.0
    assert float(q5.abs().max().item()) == 57344.0
    assert clip4 > clip5

    rep = run_hadamard_and_fp8_benchmark(dim=64, batch_size=64, seed=37)
    assert rep.orthogonal_invariance_max_diff < 1e-3
    assert rep.outlier_crest_reduction_x > 2.5
    assert rep.int4_mse_improvement_x > 1.5
    assert rep.fp8_e5m2_sqnr_db > rep.fp8_e4m3_sqnr_db


def test_roofline_and_ollama_audit_mocked_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="Unknown accelerator"):
        analyze_hardware_roofline("unknown-chip")

    for chip in ("h100-sxm5", "m3-max", "a100-sxm4"):
        r = analyze_hardware_roofline(chip)
        assert len(r.points) == 6
        assert r.points[2].speedup_vs_fp16_x > 3.0

    out_path = tmp_path / "quant_eval.json"

    def fake_get(url: str, timeout: float = 2.5) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3.5:4b",
                        "size": int(3.16 * (1024**3)),
                        "details": {"quantization_level": "Q4_K_M"},
                    }
                ]
            },
            request=httpx.Request("GET", url),
        )

    expected_answers = [9, 64, 200, 448, 6, 361]
    idx_box = {"i": 0}

    def fake_post(url: str, json: dict[str, Any], timeout: float = 45.0) -> httpx.Response:
        ans = expected_answers[idx_box["i"] % len(expected_answers)]
        idx_box["i"] += 1
        return httpx.Response(
            200,
            json={
                "response": f'{{"scratchpad": "ok", "answer": {ans}}}',
                "eval_count": 40,
                "eval_duration": 500_000_000,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    live_rep = run_real_ollama_quant_audit(save_path=out_path, allow_offline_replay=False)
    assert live_rep.accuracy_pct == 100.0
    assert live_rep.mean_decode_tokens_per_sec == 80.0
    assert out_path.exists()

    def offline_get(url: str, timeout: float = 2.5) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline_get)
    replay = run_real_ollama_quant_audit(save_path=out_path, allow_offline_replay=True)
    assert replay.accuracy_count == 6

    with pytest.raises(RuntimeError, match="unreachable"):
        run_real_ollama_quant_audit(save_path=tmp_path / "none.json", allow_offline_replay=False)


async def test_fastapi_endpoints_and_cli_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/api/v1/overview")).status_code == 200
        assert (await client.get("/api/v1/int4/benchmark")).status_code == 200
        assert (
            await client.post("/api/v1/gptq/benchmark", json={"outlier_multiplier": 15.0})
        ).status_code == 200
        assert (
            await client.post("/api/v1/hadamard/benchmark", json={"dim": 64, "batch_size": 32})
        ).status_code == 200
        assert (
            await client.post("/api/v1/roofline/analyze", json={"accelerator": "m3-max"})
        ).status_code == 200
        assert (await client.get("/")).status_code == 200

    headless = create_app(ui_dist_dir=tmp_path / "no_ui")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=headless), base_url="http://test"
    ) as hc:
        res_h = await hc.get("/")
        assert res_h.json()["mode"] == "headless-api"

    runner = CliRunner()
    for args in (
        ["benchmark-int4"],
        ["benchmark-int4", "--json"],
        ["benchmark-gptq"],
        ["benchmark-gptq", "--json"],
        ["benchmark-hadamard"],
        ["benchmark-hadamard", "--json"],
        ["roofline", "--accelerator", "a100-sxm4"],
        ["roofline", "--json"],
        ["demo"],
        ["demo", "--json"],
    ):
        res = runner.invoke(cli_app, args)
        assert res.exit_code == 0

    saved = tmp_path / "saved_q.json"
    saved.write_text(
        json.dumps(
            {
                "model_name": "qwen3.5:4b",
                "quantization_level": "Q4_K_M",
                "physical_model_size_gb": 3.16,
                "estimated_fp16_size_gb": 8.60,
                "memory_footprint_reduction_x": 2.72,
                "accuracy_count": 1,
                "num_tasks": 1,
                "accuracy_pct": 100.0,
                "mean_decode_tokens_per_sec": 82.0,
                "outcomes": [
                    {
                        "task_id": "QF-01",
                        "topic": "Test",
                        "expected_value": 9,
                        "predicted_value": 9,
                        "exact_match": True,
                        "eval_tokens": 30,
                        "tokens_per_sec": 82.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def offline_get(url: str, timeout: float = 2.5) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline_get)
    assert runner.invoke(cli_app, ["eval-ollama", "--save-path", str(saved)]).exit_code == 0
    assert (
        runner.invoke(cli_app, ["eval-ollama", "--save-path", str(saved), "--json"]).exit_code == 0
    )

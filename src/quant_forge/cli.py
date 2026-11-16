"""Rich & JSON command-line interface for QuantForge (`quant`)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from quant_forge.api.app import create_app
from quant_forge.kernels.native_int4 import benchmark_packed_int4_kernel
from quant_forge.quantizers.gptq_awq import run_gptq_vs_awq_benchmark
from quant_forge.roofline.profiler import analyze_hardware_roofline, run_real_ollama_quant_audit
from quant_forge.transforms.hadamard_fp8 import run_hadamard_and_fp8_benchmark

app = typer.Typer(
    name="quant",
    help="QuantForge: Sub-Byte LLM Compression, Second-Order GPTQ/AWQ, SpinQuant & Native C++20 INT4 Engine",
    add_completion=False,
)
console = Console()


@app.command("benchmark-int4")
def benchmark_int4_cmd(
    group_size: int = typer.Option(64, help="Even group size dividing in_features"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Benchmark native C++20 packed INT4 (2 weights/byte) storage & fused Dequant-GEMM."""
    rep = benchmark_packed_int4_kernel(group_size=group_size)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(title=f"QuantForge Native Packed INT4 Kernel ({rep.kernel_backend})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="green")
    tbl.add_row(
        "FP32 vs Packed INT4 Bytes",
        f"{rep.fp32_bytes:,} B -> {rep.packed_total_bytes:,} B ({rep.compression_ratio_x:.2f}x)",
    )
    tbl.add_row(
        "Effective Bits / Weight (incl. Scale/Zero)", f"{rep.effective_bits_per_weight:.2f} bits"
    )
    tbl.add_row(
        "Fused C++20 vs Explicit Dequant Max |Delta|", f"{rep.cpp_vs_dequant_max_abs_diff:.8f}"
    )
    tbl.add_row("Output Reconstruction SQNR", f"{rep.output_sqnr_db:.2f} dB")
    console.print(tbl)


@app.command("benchmark-gptq")
def benchmark_gptq_cmd(
    outlier_mult: float = typer.Option(
        22.0, help="Heavy-tailed activation outlier channel multiplier"
    ),
    seed: int = typer.Option(29, help="Deterministic RNG seed"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Compare Naive RTN vs MLSys'24 AWQ vs ICLR'23 Cholesky GPTQ under activation outliers."""
    rep = run_gptq_vs_awq_benchmark(outlier_multiplier=outlier_mult, seed=seed)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(
        title=f"QuantForge 4-Bit Calibration Benchmark (Outlier={rep.outlier_channel_multiplier}x, AWQ gamma*={rep.optimal_awq_gamma})"
    )
    tbl.add_column("Quantizer", style="cyan")
    tbl.add_column("Output MSE", justify="right")
    tbl.add_column("SQNR (dB)", justify="right", style="green")
    tbl.add_column("MSE Reduction vs RTN", justify="right", style="magenta")
    for m in (rep.rtn_metrics, rep.awq_metrics, rep.gptq_metrics):
        tbl.add_row(
            m.method,
            f"{m.output_mse:.7f}",
            f"{m.sqnr_db:.2f} dB",
            f"{m.relative_error_reduction_vs_rtn_x:.2f}x",
        )
    console.print(tbl)


@app.command("benchmark-hadamard")
def benchmark_hadamard_cmd(
    dim: int = typer.Option(128, help="Power-of-two channel dimension"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Verify Walsh-Hadamard (SpinQuant/QuaRot) rotation invariance, outlier crushing & OCP FP8."""
    rep = run_hadamard_and_fp8_benchmark(dim=dim)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(title=f"QuantForge Walsh-Hadamard Incoherence & OCP FP8 (D={rep.dimension})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="green")
    tbl.add_row(
        "Orthogonal Invariance |(XH)(WH)^T - XW^T|_max", f"{rep.orthogonal_invariance_max_diff:.7f}"
    )
    tbl.add_row(
        "Activation Crest Factor (Pre -> Post FWHT)",
        f"{rep.pre_rotation_crest_factor:.2f} -> {rep.post_rotation_crest_factor:.2f} ({rep.outlier_crest_reduction_x:.2f}x crushed)",
    )
    tbl.add_row(
        "INT4 Output MSE (Unrotated -> Hadamard Rotated)",
        f"{rep.unrotated_int4_mse:.6f} -> {rep.hadamard_rotated_int4_mse:.6f} ({rep.int4_mse_improvement_x:.2f}x better)",
    )
    tbl.add_row(
        "Gradient FP8 SQNR: E4M3 vs E5M2",
        f"{rep.fp8_e4m3_sqnr_db:.2f} dB (clip {rep.fp8_overflow_clipped_e4m3_pct:.2f}%) vs {rep.fp8_e5m2_sqnr_db:.2f} dB (clip {rep.fp8_overflow_clipped_e5m2_pct:.2f}%)",
    )
    console.print(tbl)


@app.command("roofline")
def roofline_cmd(
    accelerator: str = typer.Option("h100-sxm5", help="h100-sxm5 | m3-max | a100-sxm4"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Analyze Williams-Waterman-Patterson Hardware Roofline across FP16, FP8, and INT4-GPTQ."""
    rep = analyze_hardware_roofline(accelerator=accelerator)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(
        title=f"Hardware Roofline: {rep.accelerator_name} (Ridge = {rep.ridge_point_flops_per_byte} FLOPs/B)"
    )
    tbl.add_column("Regime", style="cyan")
    tbl.add_column("Precision")
    tbl.add_column("AI (FLOPs/B)", justify="right")
    tbl.add_column("Attainable TFLOP/s", justify="right", style="green")
    tbl.add_column("Bottleneck")
    tbl.add_column("Speedup", justify="right", style="magenta")
    for p in rep.points:
        tbl.add_row(
            p.regime,
            p.precision,
            f"{p.arithmetic_intensity_flops_per_byte:.2f}",
            f"{p.attainable_tflops:.2f}",
            p.bottleneck,
            f"{p.speedup_vs_fp16_x:.2f}x",
        )
    console.print(tbl)


@app.command("eval-ollama")
def eval_ollama_cmd(
    model: str = typer.Option("qwen3.5:4b", help="Ollama quantized model tag"),
    live_only: bool = typer.Option(False, "--live-only", help="Require live Ollama inference"),
    save_path: Path | None = typer.Option(None, help="Optional custom output JSON path"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Audit live quantized Ollama model physical footprint, decode tok/s & systems-math accuracy."""
    rep = run_real_ollama_quant_audit(
        model=model,
        save_path=save_path,
        allow_offline_replay=not live_only,
    )
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(
        title=(
            f"Zero-Mock Quantized Ollama Audit ({rep.model_name} [{rep.quantization_level}] • "
            f"{rep.estimated_fp16_size_gb:.2f} GB -> {rep.physical_model_size_gb:.2f} GB [{rep.memory_footprint_reduction_x:.2f}x])"
        )
    )
    tbl.add_column("Task ID", style="cyan")
    tbl.add_column("Expected", justify="right")
    tbl.add_column("Predicted", justify="right")
    tbl.add_column("Decode Speed", justify="right", style="green")
    for o in rep.outcomes:
        p_str = (
            f"[green]{o.predicted_value} ✓[/]"
            if o.exact_match
            else f"[red]{o.predicted_value} ✗[/]"
        )
        tbl.add_row(o.task_id, str(o.expected_value), p_str, f"{o.tokens_per_sec:.1f} tok/s")
    console.print(tbl)
    console.print(
        f"[bold]Summary:[/] Accuracy = {rep.accuracy_count}/{rep.num_tasks} ({rep.accuracy_pct:.2f}%) | "
        f"Mean Decode Speed = {rep.mean_decode_tokens_per_sec:.1f} tok/s"
    )


@app.command("demo")
def demo_cmd(
    as_json: bool = typer.Option(False, "--json", help="Emit unified JSON report"),
) -> None:
    """Run all deterministic QuantForge subsystems end-to-end."""
    p_int4 = benchmark_packed_int4_kernel()
    p_gptq = run_gptq_vs_awq_benchmark()
    p_had = run_hadamard_and_fp8_benchmark()
    p_roof = analyze_hardware_roofline("h100-sxm5")
    payload = {
        "packed_int4": p_int4.model_dump(),
        "gptq_vs_awq": p_gptq.model_dump(),
        "hadamard_fp8": p_had.model_dump(),
        "roofline": p_roof.model_dump(),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"[bold cyan]1. Packed INT4 ({p_int4.kernel_backend}):[/] {p_int4.compression_ratio_x:.2f}x compression ({p_int4.effective_bits_per_weight:.2f} bpw), |Delta|={p_int4.cpp_vs_dequant_max_abs_diff:.8f}"
    )
    console.print(
        f"[bold magenta]2. GPTQ vs AWQ vs RTN:[/] AWQ={p_gptq.awq_metrics.relative_error_reduction_vs_rtn_x:.2f}x, GPTQ={p_gptq.gptq_metrics.relative_error_reduction_vs_rtn_x:.2f}x lower MSE than RTN"
    )
    console.print(
        f"[bold green]3. Walsh-Hadamard & FP8:[/] Crest {p_had.outlier_crest_reduction_x:.2f}x crushed, INT4 MSE {p_had.int4_mse_improvement_x:.2f}x better, E5M2 SQNR {p_had.fp8_e5m2_sqnr_db:.2f} dB"
    )
    console.print(
        f"[bold yellow]4. Roofline ({p_roof.accelerator_name}):[/] INT4 Decode Speedup = {p_roof.points[2].speedup_vs_fp16_x:.2f}x"
    )


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8050, help="Bind port"),
) -> None:
    """Launch the QuantForge FastAPI & Sub-Byte Compression Workbench server."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()

"""Typed domain models for QuantForge compression, GPTQ/AWQ, Hadamard/FP8, and Roofline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PackedINT4BenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    kernel_backend: str
    out_features: int
    in_features: int
    group_size: int
    fp32_bytes: int
    packed_total_bytes: int
    compression_ratio_x: float
    effective_bits_per_weight: float
    cpp_vs_dequant_max_abs_diff: float
    output_sqnr_db: float


class QuantMethodMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    bits: int
    output_mse: float
    sqnr_db: float
    relative_error_reduction_vs_rtn_x: float


class GPTQvsAWQBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    out_features: int
    in_features: int
    group_size: int
    outlier_channel_multiplier: float
    optimal_awq_gamma: float
    hessian_damping_pct: float
    rtn_metrics: QuantMethodMetrics
    awq_metrics: QuantMethodMetrics
    gptq_metrics: QuantMethodMetrics


class HadamardFP8Report(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: int
    orthogonal_invariance_max_diff: float
    pre_rotation_crest_factor: float
    post_rotation_crest_factor: float
    outlier_crest_reduction_x: float
    unrotated_int4_mse: float
    hadamard_rotated_int4_mse: float
    int4_mse_improvement_x: float
    fp8_e4m3_sqnr_db: float
    fp8_e5m2_sqnr_db: float
    fp8_overflow_clipped_e4m3_pct: float = Field(ge=0.0, le=100.0)
    fp8_overflow_clipped_e5m2_pct: float = Field(ge=0.0, le=100.0)


class RooflineRegimePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    regime: str
    precision: str
    arithmetic_intensity_flops_per_byte: float
    attainable_tflops: float
    bottleneck: str
    speedup_vs_fp16_x: float


class HardwareRooflineReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    accelerator_name: str
    peak_fp16_tflops: float
    hbm_bandwidth_gbps: float
    ridge_point_flops_per_byte: float
    points: list[RooflineRegimePoint]


class RealQuantTaskOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    topic: str
    expected_value: int
    predicted_value: int | None
    exact_match: bool
    eval_tokens: int
    tokens_per_sec: float


class RealOllamaQuantReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    quantization_level: str
    physical_model_size_gb: float
    estimated_fp16_size_gb: float
    memory_footprint_reduction_x: float
    accuracy_count: int
    num_tasks: int
    accuracy_pct: float
    mean_decode_tokens_per_sec: float
    outcomes: list[RealQuantTaskOutcome]

import React, { useEffect, useState } from "react";

interface PackedINT4Report {
  kernel_backend: string;
  out_features: number;
  in_features: number;
  group_size: number;
  fp32_bytes: number;
  packed_total_bytes: number;
  compression_ratio_x: number;
  effective_bits_per_weight: number;
  cpp_vs_dequant_max_abs_diff: number;
  output_sqnr_db: number;
}

interface QuantMethod {
  method: string;
  bits: number;
  output_mse: number;
  sqnr_db: number;
  relative_error_reduction_vs_rtn_x: number;
}

interface GPTQReport {
  out_features: number;
  in_features: number;
  group_size: number;
  outlier_channel_multiplier: number;
  optimal_awq_gamma: number;
  hessian_damping_pct: number;
  rtn_metrics: QuantMethod;
  awq_metrics: QuantMethod;
  gptq_metrics: QuantMethod;
}

interface HadamardReport {
  dimension: number;
  orthogonal_invariance_max_diff: number;
  pre_rotation_crest_factor: number;
  post_rotation_crest_factor: number;
  outlier_crest_reduction_x: number;
  unrotated_int4_mse: number;
  hadamard_rotated_int4_mse: number;
  int4_mse_improvement_x: number;
  fp8_e4m3_sqnr_db: number;
  fp8_e5m2_sqnr_db: number;
  fp8_overflow_clipped_e4m3_pct: number;
  fp8_overflow_clipped_e5m2_pct: number;
}

interface RooflinePoint {
  regime: string;
  precision: string;
  arithmetic_intensity_flops_per_byte: number;
  attainable_tflops: number;
  bottleneck: string;
  speedup_vs_fp16_x: number;
}

interface RooflineReport {
  accelerator_name: string;
  peak_fp16_tflops: number;
  hbm_bandwidth_gbps: number;
  ridge_point_flops_per_byte: number;
  points: RooflinePoint[];
}

interface QuantTaskOutcome {
  task_id: string;
  topic: string;
  expected_value: number;
  predicted_value: number | null;
  exact_match: boolean;
  eval_tokens: number;
  tokens_per_sec: number;
}

interface RealQuantReport {
  model_name: string;
  quantization_level: string;
  physical_model_size_gb: number;
  estimated_fp16_size_gb: number;
  memory_footprint_reduction_x: number;
  accuracy_count: number;
  num_tasks: number;
  accuracy_pct: number;
  mean_decode_tokens_per_sec: number;
  outcomes: QuantTaskOutcome[];
}

interface OverviewPayload {
  kernel_backend: string;
  packed_int4: PackedINT4Report;
  gptq_vs_awq: GPTQReport;
  hadamard_fp8: HadamardReport;
  roofline: RooflineReport;
  real_qwen_quant: RealQuantReport | null;
}

export function App(): React.JSX.Element {
  const [data, setData] = useState<OverviewPayload | null>(null);
  const [gptq, setGptq] = useState<GPTQReport | null>(null);
  const [roof, setRoof] = useState<RooflineReport | null>(null);
  const [outlierMult, setOutlierMult] = useState<number>(22);
  const [accel, setAccel] = useState<string>("h100-sxm5");

  useEffect(() => {
    fetch("/api/v1/overview")
      .then((r) => r.json())
      .then((payload: OverviewPayload) => {
        setData(payload);
        setGptq(payload.gptq_vs_awq);
        setRoof(payload.roofline);
      });
  }, []);

  const recalibrateGptq = async () => {
    const r = await fetch("/api/v1/gptq/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outlier_multiplier: outlierMult }),
    });
    setGptq((await r.json()) as GPTQReport);
  };

  const changeAccel = async (nextAccel: string) => {
    setAccel(nextAccel);
    const r = await fetch("/api/v1/roofline/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accelerator: nextAccel }),
    });
    setRoof((await r.json()) as RooflineReport);
  };

  if (!data || !gptq || !roof) {
    return (
      <div className="shell">
        <h2>Initializing QuantForge Sub-Byte Quantization &amp; Roofline Workbench...</h2>
      </div>
    );
  }

  const bestRed = Math.max(
    gptq.awq_metrics.relative_error_reduction_vs_rtn_x,
    gptq.gptq_metrics.relative_error_reduction_vs_rtn_x,
  );
  const qwen = data.real_qwen_quant;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <h1>QuantForge — Sub-Byte LLM Quantization, Hessian GPTQ/AWQ &amp; Roofline Engine</h1>
          <p>
            Native C++20 Bit-Packed INT4 Fused GEMM · Cholesky H⁻¹ GPTQ &amp; Salient AWQ ·
            QuaRot/SpinQuant Walsh-Hadamard FWHT · OCP FP8 E4M3/E5M2
          </p>
        </div>
        <div className="pill-row">
          <span className="pill lime">{data.kernel_backend}</span>
          <span className="pill">2 Weights / Byte (INT4)</span>
          <span className="pill">Zero-Materialization Register GEMM</span>
        </div>
      </header>

      <section className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Packed INT4 Memory Compression</div>
          <div className="kpi-value">{data.packed_int4.compression_ratio_x.toFixed(2)}× Smaller</div>
          <div className="kpi-sub">
            {data.packed_int4.effective_bits_per_weight.toFixed(2)} eff bits/w · C++20 |Δ| ={" "}
            {data.packed_int4.cpp_vs_dequant_max_abs_diff.toFixed(7)}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">GPTQ / AWQ Outlier Error Reduction</div>
          <div className="kpi-value">{bestRed.toFixed(2)}× Lower MSE</div>
          <div className="kpi-sub">
            SQNR: {gptq.rtn_metrics.sqnr_db.toFixed(1)} dB →{" "}
            {Math.max(gptq.awq_metrics.sqnr_db, gptq.gptq_metrics.sqnr_db).toFixed(1)} dB (AWQ γ*=
            {gptq.optimal_awq_gamma})
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Walsh-Hadamard SpinQuant Rotation</div>
          <div className="kpi-value">{data.hadamard_fp8.int4_mse_improvement_x.toFixed(2)}× INT4 Gain</div>
          <div className="kpi-sub">
            Crest {data.hadamard_fp8.pre_rotation_crest_factor.toFixed(1)} →{" "}
            {data.hadamard_fp8.post_rotation_crest_factor.toFixed(1)} ({data.hadamard_fp8.outlier_crest_reduction_x.toFixed(2)}× crushed)
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Live Qwen-3.5-4B Quantized Audit</div>
          <div className="kpi-value">
            {qwen ? `${qwen.mean_decode_tokens_per_sec.toFixed(1)} tok/s` : "Pending"}
          </div>
          <div className="kpi-sub">
            {qwen
              ? `${qwen.quantization_level} (${qwen.physical_model_size_gb} GB · ${qwen.memory_footprint_reduction_x}×) · ${qwen.accuracy_pct.toFixed(0)}% Exact`
              : "Run quant eval-ollama"}
          </div>
        </div>
      </section>

      <section className="main-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>1. Second-Order Cholesky GPTQ vs. MLSys &apos;24 AWQ vs. RTN</h2>
            <div className="controls">
              <span style={{ fontSize: "0.76rem", color: "#8ea5be" }}>Outlier ×</span>
              <input
                type="number"
                min={4}
                max={50}
                style={{ width: 56 }}
                value={outlierMult}
                onChange={(e) => setOutlierMult(Number(e.target.value))}
              />
              <button className="btn" onClick={recalibrateGptq}>
                Calibrate H⁻¹
              </button>
            </div>
          </div>

          <table className="metric-table">
            <thead>
              <tr>
                <th>4-Bit Quantizer</th>
                <th>Output MSE</th>
                <th>SQNR (dB)</th>
                <th>Gain vs. RTN</th>
              </tr>
            </thead>
            <tbody>
              {[gptq.rtn_metrics, gptq.awq_metrics, gptq.gptq_metrics].map((m) => (
                <tr key={m.method}>
                  <td className="mono">{m.method}</td>
                  <td className="mono">{m.output_mse.toFixed(6)}</td>
                  <td className={`mono ${m.sqnr_db >= 24 ? "ok" : "amber"}`}>
                    {m.sqnr_db.toFixed(2)} dB
                  </td>
                  <td className={`mono ${m.relative_error_reduction_vs_rtn_x > 1.5 ? "ok" : ""}`}>
                    {m.relative_error_reduction_vs_rtn_x.toFixed(2)}×
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="sub-divider" />

          <div className="panel-header" style={{ marginBottom: 6 }}>
            <h2>2. QuaRot / SpinQuant Walsh-Hadamard &amp; OCP FP8 (E4M3 vs. E5M2)</h2>
          </div>
          <table className="metric-table">
            <tbody>
              <tr>
                <td>Exact FWHT Orthogonality |(XH)(WH)ᵀ - XWᵀ|_max</td>
                <td className="mono ok">
                  {data.hadamard_fp8.orthogonal_invariance_max_diff.toFixed(7)}
                </td>
              </tr>
              <tr>
                <td>Outlier Crest Factor (Unrotated → FWHT Rotated)</td>
                <td className="mono ok">
                  {data.hadamard_fp8.pre_rotation_crest_factor.toFixed(2)} →{" "}
                  {data.hadamard_fp8.post_rotation_crest_factor.toFixed(2)} (
                  {data.hadamard_fp8.outlier_crest_reduction_x.toFixed(2)}×)
                </td>
              </tr>
              <tr>
                <td>INT4 Output MSE (Unrotated → Hadamard Incoherent)</td>
                <td className="mono ok">
                  {data.hadamard_fp8.unrotated_int4_mse.toFixed(6)} →{" "}
                  {data.hadamard_fp8.hadamard_rotated_int4_mse.toFixed(6)} (
                  {data.hadamard_fp8.int4_mse_improvement_x.toFixed(2)}×)
                </td>
              </tr>
              <tr>
                <td>Backward Spike Gradient SQNR: FP8 E4M3 vs. FP8 E5M2</td>
                <td className="mono">
                  <span className="warn">
                    {data.hadamard_fp8.fp8_e4m3_sqnr_db.toFixed(1)} dB (clip{" "}
                    {data.hadamard_fp8.fp8_overflow_clipped_e4m3_pct.toFixed(2)}%)
                  </span>{" "}
                  →{" "}
                  <span className="ok">
                    {data.hadamard_fp8.fp8_e5m2_sqnr_db.toFixed(1)} dB (clip{" "}
                    {data.hadamard_fp8.fp8_overflow_clipped_e5m2_pct.toFixed(2)}%)
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>3. Hardware Roofline Autotuner (Ridge = {roof.ridge_point_flops_per_byte} FLOPs/B)</h2>
            <div className="controls">
              <select value={accel} onChange={(e) => changeAccel(e.target.value)}>
                <option value="h100-sxm5">NVIDIA H100 SXM5 80GB</option>
                <option value="m3-max">Apple M3 Max 128GB</option>
                <option value="a100-sxm4">NVIDIA A100 SXM4 80GB</option>
              </select>
            </div>
          </div>

          <table className="metric-table">
            <thead>
              <tr>
                <th>Workload</th>
                <th>Precision</th>
                <th>AI (F/B)</th>
                <th>TFLOP/s</th>
                <th>Bound</th>
                <th>Speedup</th>
              </tr>
            </thead>
            <tbody>
              {roof.points.map((p, i) => (
                <tr key={i}>
                  <td>{p.regime}</td>
                  <td className="mono">{p.precision}</td>
                  <td className="mono">{p.arithmetic_intensity_flops_per_byte.toFixed(2)}</td>
                  <td className="mono">{p.attainable_tflops.toFixed(1)}</td>
                  <td className={p.bottleneck === "MEMORY-BANDWIDTH" ? "amber" : "ok"}>
                    {p.bottleneck}
                  </td>
                  <td className="mono ok">{p.speedup_vs_fp16_x.toFixed(2)}×</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="sub-divider" />
          <div style={{ fontSize: "0.79rem", color: "#8ea5be", lineHeight: 1.45 }}>
            <strong>Packed C++20 Storage Proof ({data.packed_int4.out_features}×{data.packed_int4.in_features}, G={data.packed_int4.group_size}):</strong>{" "}
            Dense FP32 occupies <span className="mono">{data.packed_int4.fp32_bytes.toLocaleString()} B</span> vs.
            bit-packed uint8 + group scales/zeros{" "}
            <span className="mono ok">{data.packed_int4.packed_total_bytes.toLocaleString()} B</span> (
            {data.packed_int4.output_sqnr_db.toFixed(2)} dB SQNR, zero FP32 weight-matrix materialization).
          </div>
        </div>
      </section>

      {qwen && (
        <section className="panel">
          <div className="panel-header">
            <h2>
              4. Zero-Mock Live Ollama ({qwen.model_name} [{qwen.quantization_level}]) Footprint,
              Bandwidth &amp; Systems-Math Audit
            </h2>
            <span className="pill lime">
              {qwen.estimated_fp16_size_gb.toFixed(2)} GB FP16 → {qwen.physical_model_size_gb.toFixed(2)} GB{" "}
              {qwen.quantization_level} ({qwen.memory_footprint_reduction_x.toFixed(2)}× Smaller ·{" "}
              {qwen.accuracy_count}/{qwen.num_tasks} Exact)
            </span>
          </div>
          <table className="metric-table">
            <thead>
              <tr>
                <th>Task ID</th>
                <th>Hardware / Compression Topic</th>
                <th>Expected</th>
                <th>Predicted</th>
                <th>Tokens</th>
                <th>Measured Decode Speed</th>
              </tr>
            </thead>
            <tbody>
              {qwen.outcomes.map((o) => (
                <tr key={o.task_id}>
                  <td className="mono">{o.task_id}</td>
                  <td>{o.topic}</td>
                  <td className="mono">{o.expected_value}</td>
                  <td className={`mono ${o.exact_match ? "ok" : "warn"}`}>
                    {o.predicted_value ?? "None"} {o.exact_match ? "✓" : "✗"}
                  </td>
                  <td className="mono">{o.eval_tokens}</td>
                  <td className="mono ok">{o.tokens_per_sec.toFixed(1)} tok/s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

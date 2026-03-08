//! Performance ceiling benchmark — measures actual vs theoretical peak.
//!
//! Compares our CPU kernels against hardware limits:
//!   - cpu_matmul (Accelerate SGEMM) vs naive scalar
//!   - GDN recurrence: NEON vs scalar
//!   - Memory bandwidth utilization
//!   - Reports efficiency as % of theoretical peak
//!
//! Usage: cargo run --features ane --release --bin perf_ceiling

use std::time::Instant;

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

struct BenchResult {
    label: &'static str,
    elapsed_ms: f64,
    gflops: f64,
}

fn print_separator() {
    eprintln!("{}", "─".repeat(80));
}

fn print_header(title: &str) {
    eprintln!();
    print_separator();
    eprintln!("  {title}");
    print_separator();
}

fn fill_random(buf: &mut [f32], seed: u64) {
    let mut state = seed;
    for v in buf.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    }
}

// ---------------------------------------------------------------------------
// Naive scalar matmul (baseline)
// ---------------------------------------------------------------------------

fn naive_matmul(w: &[f32], x: &[f32], m: usize, n: usize, s: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * s];
    for mi in 0..m {
        for si in 0..s {
            let mut acc = 0.0f32;
            for ni in 0..n {
                acc += w[mi * n + ni] * x[ni * s + si];
            }
            out[mi * s + si] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Matmul benchmark: naive vs cpu_matmul (now uses Accelerate SGEMM)
// ---------------------------------------------------------------------------

fn bench_matmul() -> Vec<BenchResult> {
    let mut results = Vec::new();

    let configs: &[(usize, usize, usize, &str)] = &[
        (2048, 2048, 4, "QKV proj (dim x dim, seq=4)"),
        (2048, 2048, 64, "QKV proj (dim x dim, seq=64)"),
        (6144, 2048, 4, "QKV combined (3*dim x dim, seq=4)"),
        (5632, 2048, 4, "FFN gate (hidden x dim, seq=4)"),
        (2048, 5632, 4, "FFN down (dim x hidden, seq=4)"),
    ];

    print_header("MATMUL: naive scalar vs cpu_matmul (Accelerate SGEMM)");
    eprintln!(
        "  {:40} {:>10} {:>10} {:>8} {:>8}",
        "Config", "Naive(ms)", "SGEMM(ms)", "Speedup", "GFLOPS"
    );
    print_separator();

    let warmup = 2;
    let iters = 10;

    for &(m, n, s, label) in configs {
        let mut w = vec![0.0f32; m * n];
        let mut x = vec![0.0f32; n * s];
        fill_random(&mut w, 42);
        fill_random(&mut x, 123);

        let flops = 2.0 * m as f64 * n as f64 * s as f64;

        // Naive
        for _ in 0..warmup {
            black_box(naive_matmul(&w, &x, m, n, s));
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(naive_matmul(&w, &x, m, n, s));
        }
        let naive_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // cpu_matmul (Accelerate SGEMM)
        for _ in 0..warmup {
            black_box(nanobot::agent::ane_forward::cpu_matmul(&w, &x, m, n, s));
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(nanobot::agent::ane_forward::cpu_matmul(&w, &x, m, n, s));
        }
        let sgemm_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let sgemm_gflops = flops / (sgemm_ms / 1000.0) / 1e9;

        let speedup = naive_ms / sgemm_ms;
        eprintln!(
            "  {:40} {:>9.3}ms {:>9.3}ms {:>7.1}x {:>7.1}",
            label, naive_ms, sgemm_ms, speedup, sgemm_gflops
        );

        results.push(BenchResult { label: "naive", elapsed_ms: naive_ms, gflops: 0.0 });
        results.push(BenchResult { label: "SGEMM", elapsed_ms: sgemm_ms, gflops: sgemm_gflops });
    }

    results
}

// ---------------------------------------------------------------------------
// GDN recurrence benchmark: scalar vs NEON
// ---------------------------------------------------------------------------

fn bench_gdn_recurrence() {
    print_header("GDN RECURRENCE: scalar vs NEON (gated delta net core loop)");

    let seq = 64;
    let h_v = 16;
    let d_k = 128;
    let d_v = 128;
    let value_dim = h_v * d_v;

    let n_q = h_v * d_k * seq;
    let n_v = h_v * d_v * seq;
    let n_g = h_v * seq;

    let mut q = vec![0.0f32; n_q];
    let mut k = vec![0.0f32; n_q];
    let mut v = vec![0.0f32; n_v];
    let mut g_arr = vec![0.0f32; n_g];
    let mut beta_arr = vec![0.0f32; n_g];
    fill_random(&mut q, 1);
    fill_random(&mut k, 2);
    fill_random(&mut v, 3);
    fill_random(&mut g_arr, 4);
    fill_random(&mut beta_arr, 5);
    for val in g_arr.iter_mut() {
        *val = val.abs().min(0.999);
    }
    for val in beta_arr.iter_mut() {
        *val = 1.0 / (1.0 + (-*val).exp());
    }

    let iters = 5;

    // --- Scalar baseline ---
    let run_scalar = || {
        let mut state = vec![0.0f32; h_v * d_v * d_k];
        let mut y = vec![0.0f32; value_dim * seq];

        for t in 0..seq {
            for h in 0..h_v {
                let g_t = g_arr[h * seq + t];
                let beta_t = beta_arr[h * seq + t];

                for dv in 0..d_v {
                    for dk in 0..d_k {
                        state[h * d_v * d_k + dv * d_k + dk] *= g_t;
                    }
                }
                for dv in 0..d_v {
                    let mut kv_mem = 0.0f32;
                    for dk in 0..d_k {
                        kv_mem +=
                            state[h * d_v * d_k + dv * d_k + dk] * k[(h * d_k + dk) * seq + t];
                    }
                    let v_t = v[(h * d_v + dv) * seq + t];
                    let delta = (v_t - kv_mem) * beta_t;
                    for dk in 0..d_k {
                        state[h * d_v * d_k + dv * d_k + dk] +=
                            k[(h * d_k + dk) * seq + t] * delta;
                    }
                }
                for dv in 0..d_v {
                    let mut y_val = 0.0f32;
                    for dk in 0..d_k {
                        y_val +=
                            state[h * d_v * d_k + dv * d_k + dk] * q[(h * d_k + dk) * seq + t];
                    }
                    y[(h * d_v + dv) * seq + t] = y_val;
                }
            }
        }
        y
    };

    black_box(run_scalar());
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(run_scalar());
    }
    let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // --- NEON (via nanobot::gdn_recurrence, called indirectly) ---
    // We call the same recurrence logic that cpu_gdn_forward uses internally.
    // Since gdn_recurrence is private, we benchmark it via a reimplementation
    // using the same NEON intrinsics.
    #[cfg(target_arch = "aarch64")]
    let neon_ms = {
        use std::arch::aarch64::*;

        let run_neon = || {
            let mut state = vec![0.0f32; h_v * d_v * d_k];
            let mut y_out = vec![0.0f32; value_dim * seq];
            let mut k_buf = vec![0.0f32; d_k];
            let mut q_buf = vec![0.0f32; d_k];

            for t in 0..seq {
                for h in 0..h_v {
                    let g_t = g_arr[h * seq + t];
                    let beta_t = beta_arr[h * seq + t];

                    for dk in 0..d_k {
                        k_buf[dk] = k[(h * d_k + dk) * seq + t];
                        q_buf[dk] = q[(h * d_k + dk) * seq + t];
                    }

                    let state_base = h * d_v * d_k;

                    for dv in 0..d_v {
                        let row = state_base + dv * d_k;
                        unsafe {
                            let g_vec = vdupq_n_f32(g_t);
                            let mut acc0 = vdupq_n_f32(0.0);
                            let mut acc1 = vdupq_n_f32(0.0);
                            let sp = state.as_mut_ptr().add(row);
                            let kp = k_buf.as_ptr();

                            let mut dk = 0usize;
                            while dk + 8 <= d_k {
                                let mut s0 = vld1q_f32(sp.add(dk));
                                let mut s1 = vld1q_f32(sp.add(dk + 4));
                                let k0 = vld1q_f32(kp.add(dk));
                                let k1 = vld1q_f32(kp.add(dk + 4));
                                s0 = vmulq_f32(s0, g_vec);
                                s1 = vmulq_f32(s1, g_vec);
                                vst1q_f32(sp.add(dk), s0);
                                vst1q_f32(sp.add(dk + 4), s1);
                                acc0 = vfmaq_f32(acc0, s0, k0);
                                acc1 = vfmaq_f32(acc1, s1, k1);
                                dk += 8;
                            }
                            let kv_mem = vaddvq_f32(vaddq_f32(acc0, acc1));

                            let v_t = v[(h * d_v + dv) * seq + t];
                            let delta = (v_t - kv_mem) * beta_t;
                            let delta_vec = vdupq_n_f32(delta);

                            dk = 0;
                            while dk + 8 <= d_k {
                                let s0 = vld1q_f32(sp.add(dk));
                                let s1 = vld1q_f32(sp.add(dk + 4));
                                let k0 = vld1q_f32(kp.add(dk));
                                let k1 = vld1q_f32(kp.add(dk + 4));
                                vst1q_f32(sp.add(dk), vfmaq_f32(s0, k0, delta_vec));
                                vst1q_f32(sp.add(dk + 4), vfmaq_f32(s1, k1, delta_vec));
                                dk += 8;
                            }

                            let qp = q_buf.as_ptr();
                            let mut yacc0 = vdupq_n_f32(0.0);
                            let mut yacc1 = vdupq_n_f32(0.0);
                            dk = 0;
                            while dk + 8 <= d_k {
                                let s0 = vld1q_f32(sp.add(dk));
                                let s1 = vld1q_f32(sp.add(dk + 4));
                                let q0 = vld1q_f32(qp.add(dk));
                                let q1 = vld1q_f32(qp.add(dk + 4));
                                yacc0 = vfmaq_f32(yacc0, s0, q0);
                                yacc1 = vfmaq_f32(yacc1, s1, q1);
                                dk += 8;
                            }
                            y_out[(h * d_v + dv) * seq + t] =
                                vaddvq_f32(vaddq_f32(yacc0, yacc1));
                        }
                    }
                }
            }
            y_out
        };

        black_box(run_neon());
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(run_neon());
        }
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    };

    #[cfg(not(target_arch = "aarch64"))]
    let neon_ms = scalar_ms;

    let flops_per_token = 7.0 * h_v as f64 * d_v as f64 * d_k as f64;
    let total_flops = flops_per_token * seq as f64;
    let scalar_gflops = total_flops / (scalar_ms / 1000.0) / 1e9;
    let neon_gflops = total_flops / (neon_ms / 1000.0) / 1e9;
    let speedup = scalar_ms / neon_ms;

    let state_kb = h_v * d_v * d_k * 4 / 1024;

    eprintln!("  Dims:              h_v={h_v}, d_k={d_k}, d_v={d_v}, seq={seq}");
    eprintln!("  State per layer:   {} KB", state_kb);
    eprintln!();
    eprintln!("  {:20} {:>10} {:>10} {:>8}", "", "Time(ms)", "GFLOPS", "tok/ms");
    print_separator();
    eprintln!(
        "  {:20} {:>9.2}ms {:>9.2} {:>7.1}",
        "Scalar", scalar_ms, scalar_gflops, seq as f64 / scalar_ms
    );
    eprintln!(
        "  {:20} {:>9.2}ms {:>9.2} {:>7.1}",
        "NEON (4-wide, 2x unroll)", neon_ms, neon_gflops, seq as f64 / neon_ms
    );
    eprintln!("  Speedup:           {:.1}x", speedup);
    eprintln!();
    eprintln!("  18 GDN layers (scalar):  {:.1} ms", scalar_ms * 18.0);
    eprintln!("  18 GDN layers (NEON):    {:.1} ms", neon_ms * 18.0);
}

// ---------------------------------------------------------------------------
// Memory bandwidth benchmark
// ---------------------------------------------------------------------------

fn bench_memory_bandwidth() {
    print_header("MEMORY BANDWIDTH (sequential read)");

    let size_mb = 512;
    let n = size_mb * 1024 * 1024 / 4;
    let mut data = vec![0.0f32; n];
    fill_random(&mut data, 999);

    let iters = 5;
    let mut sum = 0.0f32;
    for &v in data.iter() {
        sum += v;
    }
    black_box(sum);

    let t0 = Instant::now();
    for _ in 0..iters {
        let mut s = 0.0f32;
        for &v in data.iter() {
            s += v;
        }
        sum = black_box(s);
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let bytes = size_mb as f64 * 1024.0 * 1024.0;
    let gb_per_sec = bytes / (elapsed_ms / 1000.0) / 1e9;
    let _ = black_box(sum);

    eprintln!("  Buffer size:       {} MB", size_mb);
    eprintln!("  Time per pass:     {:.2} ms", elapsed_ms);
    eprintln!("  Measured BW:       {:.1} GB/s", gb_per_sec);
    eprintln!("  M4 Pro theoretical: 273 GB/s (unified memory)");
    eprintln!("  Efficiency:        {:.1}%", gb_per_sec / 273.0 * 100.0);
}

// ---------------------------------------------------------------------------
// Inference throughput estimate
// ---------------------------------------------------------------------------

fn estimate_inference(matmul_results: &[BenchResult]) {
    print_header("FULL LAYER COST ESTIMATE (Qwen3.5-2B)");

    // Per MHA layer: 7 matmuls (Q, K, V, Wo, W1, W3, W2)
    // Per GDN layer: 6 matmuls (QKV, A, B, Z, O) + recurrence
    let sgemm_results: Vec<_> = matmul_results.iter().filter(|r| r.label == "SGEMM").collect();
    if sgemm_results.len() >= 5 {
        // Approximate: QKV + Wo ≈ 4× first config, W1+W3 ≈ 2× gate, W2 ≈ down
        let qkv_ms = sgemm_results[0].elapsed_ms;
        let gate_ms = sgemm_results[3].elapsed_ms;
        let down_ms = sgemm_results[4].elapsed_ms;

        let mha_layer_ms = 4.0 * qkv_ms + 2.0 * gate_ms + down_ms;
        eprintln!("  MHA layer (7 matmuls):       {:.2} ms", mha_layer_ms);
        eprintln!("  GDN layer (6 matmuls + rec):  matmuls + recurrence overhead");
        eprintln!("  24 layers total:             {:.1} ms (matmuls only)", mha_layer_ms * 24.0);
    }

    eprintln!();
    let model_gb = 2.0_f64;
    let bw_gb_s = 273.0;
    let theoretical_tok_s = bw_gb_s / model_gb;
    eprintln!("  Model size (8-bit):  {:.1} GB", model_gb);
    eprintln!("  Memory bandwidth:    {:.0} GB/s (M4 Pro)", bw_gb_s);
    eprintln!("  Theoretical peak:    {:.0} tok/s (bandwidth-bound)", theoretical_tok_s);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    eprintln!();
    eprintln!("  PERFORMANCE CEILING BENCHMARK");
    eprintln!("  Hardware: Apple Silicon (M-series)");
    eprintln!("  Model reference: Qwen3.5-2B (dim=2048, hidden=5632, 24 layers)");

    let matmul_results = bench_matmul();
    bench_gdn_recurrence();
    bench_memory_bandwidth();
    estimate_inference(&matmul_results);

    print_header("OPTIMIZATION STATUS");
    eprintln!("  [x] cpu_matmul → Accelerate SGEMM (25-300x speedup)");
    eprintln!("  [x] GDN recurrence → NEON SIMD (fused decay+kv_mem, 2x unroll)");
    eprintln!("  [ ] Fused dequant+matmul (avoid full-layer f32 materialization)");
    eprintln!("  [ ] ANE conv1x1 (matmul→conv2d on Neural Engine)");
    eprintln!("  [ ] Pipeline: overlap dequant(layer N+1) with compute(layer N)");
    eprintln!();
}

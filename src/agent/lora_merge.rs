#![allow(unsafe_code)]
//! LoRA merge-on-disk: bake trained LoRA deltas directly into base model weights.
//!
//! Sidesteps the broken oMLX adapter API by merging `δW = scale * B @ A` into
//! the quantized safetensors on disk. After merge + unload, the next oMLX request
//! loads the merged model as if it were the original.
//!
//! Quantization error compounding is prevented by always merging from the pristine
//! original weights (backed up as `*.safetensors.premrg`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use tracing::{debug, info};

use super::ane_lora::{LoraAdapter, LoraModel};
use super::ane_mlx_bridge::LoraTarget;
use super::ane_weights::{dequant_nbit, quantize_nbit};

/// Result of a merge operation.
#[derive(Debug)]
pub struct MergeReport {
    pub tensors_merged: usize,
    pub files_modified: usize,
    pub backup_created: bool,
    pub elapsed_ms: u64,
}

/// A pending LoRA delta to apply to a base weight tensor.
struct PendingDelta {
    /// Full tensor base name, e.g. `language_model.model.layers.0.self_attn.q_proj`
    base_name: String,
    adapter: DeltaAdapter,
}

struct DeltaAdapter {
    a: Vec<f32>, // [rank, d_in]
    b: Vec<f32>, // [d_out, rank]
    rank: usize,
    d_in: usize,
    d_out: usize,
    scale: f32, // alpha / rank
}

/// Parsed tensor entry within a safetensors file.
struct TensorEntry {
    #[allow(dead_code)]
    dtype: String,
    shape: Vec<usize>,
    data_start: usize, // offset within data section
    data_end: usize,
}

/// Merge trained LoRA weights into the base model's quantized safetensors.
///
/// Sequence: unload from oMLX → backup originals → dequantize → add delta →
/// requantize → write back. The next oMLX inference request auto-loads the merged model.
pub fn merge_lora_into_base(
    model_dir: &Path,
    lora: &LoraModel,
    weight_prefix: &str,
    linear_attn_indices: &[usize],
    bits: usize,
    group_size: usize,
) -> Result<MergeReport> {
    let start = std::time::Instant::now();

    // 1. Unload model from oMLX so file handles are released.
    info!("LoRA merge: unloading model from oMLX...");
    super::learn_loop::omlx_try_reload_from_config();
    std::thread::sleep(std::time::Duration::from_secs(2));

    // 2. Collect all LoRA deltas with their target tensor names.
    let scale = lora.config.alpha / lora.config.rank as f32;
    let deltas = collect_deltas(lora, weight_prefix, linear_attn_indices, scale);
    if deltas.is_empty() {
        info!("LoRA merge: no trained adapters, skipping");
        return Ok(MergeReport {
            tensors_merged: 0,
            files_modified: 0,
            backup_created: false,
            elapsed_ms: start.elapsed().as_millis() as u64,
        });
    }
    info!("LoRA merge: {} adapter deltas to apply", deltas.len());

    // 3. Index safetensors files and group deltas by file.
    let st_files = list_safetensors(model_dir)?;
    let mut deltas = deltas;
    let file_deltas = assign_deltas_to_files(&st_files, &mut deltas)?;

    let mut tensors_merged = 0usize;
    let mut files_modified = 0usize;
    let mut backup_created = false;

    // 4. Process each file that has modifications.
    for (file_path, delta_indices) in &file_deltas {
        let premrg = premrg_path(file_path);
        let source_path = if premrg.exists() {
            &premrg
        } else {
            file_path
        };

        // Backup original on first merge.
        if !premrg.exists() {
            info!(
                "LoRA merge: backing up {}",
                file_path.file_name().unwrap().to_string_lossy()
            );
            std::fs::copy(file_path, &premrg).with_context(|| {
                format!(
                    "backup {} → {}",
                    file_path.display(),
                    premrg.display()
                )
            })?;
            backup_created = true;
        }

        // Read source file (always from .premrg = pristine original).
        let mut buf = std::fs::read(source_path)
            .with_context(|| format!("read {}", source_path.display()))?;

        let (header_size, entries) = parse_safetensors_header(&buf)?;
        let data_start = 8 + header_size;

        for &di in delta_indices {
            let delta = &deltas[di];
            let n_modified = apply_delta_to_buffer(
                &mut buf,
                data_start,
                &entries,
                delta,
                bits,
                group_size,
            )?;
            tensors_merged += n_modified;
        }

        // Write modified file back to the original path.
        std::fs::write(file_path, &buf)
            .with_context(|| format!("write {}", file_path.display()))?;
        files_modified += 1;

        debug!(
            "LoRA merge: wrote {} ({} deltas applied)",
            file_path.file_name().unwrap().to_string_lossy(),
            delta_indices.len()
        );
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    info!(
        "LoRA merge: {tensors_merged} tensors in {files_modified} files, {elapsed_ms}ms"
    );

    Ok(MergeReport {
        tensors_merged,
        files_modified,
        backup_created,
        elapsed_ms,
    })
}

// ---------------------------------------------------------------------------
// Delta collection
// ---------------------------------------------------------------------------

fn collect_deltas(
    lora: &LoraModel,
    prefix: &str,
    linear_attn_indices: &[usize],
    scale: f32,
) -> Vec<PendingDelta> {
    let mut out = Vec::new();

    for (layer_idx, layer) in lora.layers.iter().enumerate() {
        let is_linear = linear_attn_indices.contains(&layer_idx);

        let targets: &[(LoraTarget, &Option<LoraAdapter>)] = &[
            (LoraTarget::QProj, &layer.wq),
            (LoraTarget::VProj, &layer.wv),
            (LoraTarget::OProj, &layer.wo),
            (LoraTarget::DownProj, &layer.w2),
        ];

        for (target, adapter) in targets {
            if is_linear && *target != LoraTarget::DownProj {
                continue;
            }
            let Some(a) = adapter else { continue };

            let proj_name = target.mlx_name();
            let attn_or_mlp = if *target == LoraTarget::DownProj {
                "mlp"
            } else {
                "self_attn"
            };
            let base_name =
                format!("{prefix}.layers.{layer_idx}.{attn_or_mlp}.{proj_name}");

            out.push(PendingDelta {
                base_name,
                adapter: DeltaAdapter {
                    a: a.a.clone(),
                    b: a.b.clone(),
                    rank: a.rank,
                    d_in: a.d_in,
                    d_out: a.d_out,
                    scale,
                },
            });
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Safetensors file handling
// ---------------------------------------------------------------------------

fn list_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map_or(false, |ext| ext == "safetensors")
                && !p
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .ends_with(".premrg.safetensors")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no safetensors files in {}", dir.display());
    }
    Ok(files)
}

fn premrg_path(file_path: &Path) -> PathBuf {
    let stem = file_path.file_stem().unwrap().to_string_lossy();
    file_path.with_file_name(format!("{stem}.premrg.safetensors"))
}

fn parse_safetensors_header(buf: &[u8]) -> Result<(usize, HashMap<String, TensorEntry>)> {
    if buf.len() < 8 {
        bail!("safetensors file too small");
    }
    let header_size = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    let hdr_json: serde_json::Value = serde_json::from_slice(&buf[8..8 + header_size])
        .context("parse safetensors header")?;

    let mut entries = HashMap::new();
    if let serde_json::Value::Object(map) = hdr_json {
        for (name, m) in &map {
            if name == "__metadata__" {
                continue;
            }
            let dtype = m["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> = m["shape"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let offsets = m["data_offsets"].as_array().unwrap();
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;

            entries.insert(
                name.clone(),
                TensorEntry {
                    dtype,
                    shape,
                    data_start: start,
                    data_end: end,
                },
            );
        }
    }

    Ok((header_size, entries))
}

/// Find which safetensors file contains each delta's target tensors.
/// Returns file_path → [delta indices].
///
/// For MoE models, `mlp.down_proj` may be stored as `mlp.shared_expert.down_proj`.
/// When the plain name isn't found, we try the `shared_expert` fallback and update
/// the delta's `base_name` so `apply_delta_to_buffer` finds the right tensor.
fn assign_deltas_to_files(
    files: &[PathBuf],
    deltas: &mut [PendingDelta],
) -> Result<Vec<(PathBuf, Vec<usize>)>> {
    // Build index: tensor_name → file_index
    let mut tensor_to_file: HashMap<String, usize> = HashMap::new();
    for (fi, path) in files.iter().enumerate() {
        let buf = std::fs::read(path)?;
        let (_, entries) = parse_safetensors_header(&buf)?;
        for name in entries.keys() {
            tensor_to_file.insert(name.clone(), fi);
        }
    }

    let mut file_deltas: Vec<Vec<usize>> = vec![Vec::new(); files.len()];
    for (di, delta) in deltas.iter_mut().enumerate() {
        let w_key = format!("{}.weight", delta.base_name);
        if let Some(&fi) = tensor_to_file.get(&w_key) {
            file_deltas[fi].push(di);
        } else {
            // MoE fallback: mlp.down_proj → mlp.shared_expert.down_proj
            let moe_name = delta.base_name.replace("mlp.", "mlp.shared_expert.");
            let moe_key = format!("{moe_name}.weight");
            if let Some(&fi) = tensor_to_file.get(&moe_key) {
                debug!(
                    "LoRA merge: {} → {} (MoE shared_expert fallback)",
                    delta.base_name, moe_name
                );
                delta.base_name = moe_name;
                file_deltas[fi].push(di);
            } else {
                bail!("tensor {} not found in any safetensors file (also tried shared_expert)", w_key);
            }
        }
    }

    Ok(files
        .iter()
        .zip(file_deltas)
        .filter(|(_, d)| !d.is_empty())
        .map(|(p, d)| (p.clone(), d))
        .collect())
}

// ---------------------------------------------------------------------------
// Delta application
// ---------------------------------------------------------------------------

/// Compute δW = scale * B @ A as a dense [d_out, d_in] matrix.
fn compute_lora_delta(adapter: &DeltaAdapter) -> Vec<f32> {
    let DeltaAdapter {
        a,
        b,
        rank,
        d_in,
        d_out,
        scale,
    } = adapter;
    let mut delta = vec![0.0f32; d_out * d_in];
    // B[o, r] * A[r, i] → delta[o, i]
    for o in 0..*d_out {
        for r in 0..*rank {
            let b_val = b[o * rank + r] * scale;
            for i in 0..*d_in {
                delta[o * d_in + i] += b_val * a[r * d_in + i];
            }
        }
    }
    delta
}

fn bf16_to_f32_slice(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bf16 = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

/// Apply a single LoRA delta to the in-memory safetensors buffer.
/// Returns the number of tensor components modified (weight + scales + biases = 1 logical tensor).
fn apply_delta_to_buffer(
    buf: &mut [u8],
    data_section_start: usize,
    entries: &HashMap<String, TensorEntry>,
    delta: &PendingDelta,
    bits: usize,
    group_size: usize,
) -> Result<usize> {
    let w_key = format!("{}.weight", delta.base_name);
    let s_key = format!("{}.scales", delta.base_name);
    let b_key = format!("{}.biases", delta.base_name);

    let w_entry = entries
        .get(&w_key)
        .ok_or_else(|| anyhow::anyhow!("missing {w_key}"))?;
    let s_entry = entries
        .get(&s_key)
        .ok_or_else(|| anyhow::anyhow!("missing {s_key}"))?;
    let b_entry = entries
        .get(&b_key)
        .ok_or_else(|| anyhow::anyhow!("missing {b_key}"))?;

    let rows = w_entry.shape[0];
    // For non-power-of-2 bits (3-bit), logical cols differ from packed shape.
    let cols = w_entry.shape[1] * 32 / bits;

    // Read current quantized weight from buffer.
    let w_abs_start = data_section_start + w_entry.data_start;
    let w_abs_end = data_section_start + w_entry.data_end;
    let s_abs_start = data_section_start + s_entry.data_start;
    let s_abs_end = data_section_start + s_entry.data_end;
    let b_abs_start = data_section_start + b_entry.data_start;
    let b_abs_end = data_section_start + b_entry.data_end;

    let w_data = &buf[w_abs_start..w_abs_end];
    let scales_f32 = bf16_to_f32_slice(&buf[s_abs_start..s_abs_end]);
    let biases_f32 = bf16_to_f32_slice(&buf[b_abs_start..b_abs_end]);

    // Dequantize base weight to f32.
    let mut base = dequant_nbit(w_data, &scales_f32, &biases_f32, rows, cols, group_size, bits);

    // Compute and add LoRA delta.
    let lora_delta = compute_lora_delta(&delta.adapter);
    if lora_delta.len() != base.len() {
        bail!(
            "shape mismatch for {}: base {}x{} = {}, delta {}x{} = {}",
            delta.base_name,
            rows,
            cols,
            base.len(),
            delta.adapter.d_out,
            delta.adapter.d_in,
            lora_delta.len()
        );
    }
    for (b, d) in base.iter_mut().zip(lora_delta.iter()) {
        *b += *d;
    }

    // Requantize with same parameters.
    let (new_w_data, new_scales, new_biases) =
        quantize_nbit(&base, rows, cols, group_size, bits);

    // Verify sizes match (same quantization params → same sizes).
    debug_assert_eq!(new_w_data.len(), w_abs_end - w_abs_start);

    // Write back into buffer.
    buf[w_abs_start..w_abs_end].copy_from_slice(&new_w_data);
    buf[s_abs_start..s_abs_end].copy_from_slice(&f32_to_bf16_bytes(&new_scales));
    buf[b_abs_start..b_abs_end].copy_from_slice(&f32_to_bf16_bytes(&new_biases));

    Ok(1)
}

// ---------------------------------------------------------------------------
// Router merge: bake trained MoE gate weights into safetensors
// ---------------------------------------------------------------------------

/// Merge trained router gate weights into the base model's safetensors on disk.
///
/// Unlike LoRA merge (which applies δW = scale * B @ A), router merge applies
/// a full-rank delta: `new_gate = original_gate + (trained - frozen)`.
///
/// The router is stored as quantized `mlp.gate` tensors in safetensors.
/// Process: dequantize → add delta → requantize → write back.
pub fn merge_router_into_safetensors(
    model_dir: &Path,
    trained_weights: &[Vec<f32>],    // per-layer [num_experts * dim]
    original_weights: &[Vec<f32>],   // frozen router weights (snapshot at init)
    num_experts: usize,
    dim: usize,
    bits: usize,
    group_size: usize,
) -> Result<MergeReport> {
    let start = std::time::Instant::now();

    // Compute per-layer deltas
    let mut layer_deltas: HashMap<usize, Vec<f32>> = HashMap::new();
    for (l, (trained, original)) in trained_weights.iter().zip(original_weights.iter()).enumerate() {
        if trained.is_empty() || original.is_empty() {
            continue; // non-MoE layer
        }
        let delta: Vec<f32> = trained.iter().zip(original.iter()).map(|(t, o)| t - o).collect();
        let max_delta = delta.iter().map(|d| d.abs()).fold(0.0f32, f32::max);
        if max_delta < 1e-8 {
            continue; // no meaningful change
        }
        debug!("Router L{l}: max_delta={max_delta:.6}");
        layer_deltas.insert(l, delta);
    }

    if layer_deltas.is_empty() {
        info!("Router merge: no layers changed, skipping");
        return Ok(MergeReport {
            tensors_merged: 0,
            files_modified: 0,
            backup_created: false,
            elapsed_ms: start.elapsed().as_millis() as u64,
        });
    }
    info!("Router merge: {} layers with deltas", layer_deltas.len());

    let st_files = list_safetensors(model_dir)?;
    let mut tensors_merged = 0usize;
    let mut files_modified = 0usize;
    let mut backup_created = false;

    for file_path in &st_files {
        // Always read from the pristine .premrg backup if it exists
        let premrg = premrg_path(file_path);
        let read_from = if premrg.exists() { &premrg } else { file_path };

        // Parse header from pristine source (mmap for >2GB)
        let raw_file = std::fs::File::open(read_from)
            .with_context(|| format!("open {}", read_from.display()))?;
        let raw = unsafe { memmap2::Mmap::map(&raw_file)
            .with_context(|| format!("mmap {}", read_from.display()))? };
        if raw.len() < 8 {
            continue;
        }
        let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let header_json: serde_json::Value =
            serde_json::from_slice(&raw[8..8 + header_len])
                .with_context(|| "parse safetensors header")?;
        let data_start = 8 + header_len;

        // Find gate tensors and their layer indices
        let mut gate_entries: Vec<(usize, String, TensorEntry)> = Vec::new();
        if let Some(obj) = header_json.as_object() {
            for (key, meta) in obj {
                if key == "__metadata__" { continue; }
                // Match keys like "model.layers.5.mlp.gate" or "language_model.model.layers.5.mlp.gate"
                if !key.contains("mlp.gate") || key.ends_with(".scales") || key.ends_with(".biases") {
                    continue;
                }
                if !key.ends_with(".weight") {
                    continue;
                }
                // Extract layer index from key
                let layer_idx = key.split("layers.")
                    .nth(1)
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                let Some(l) = layer_idx else { continue };

                if !layer_deltas.contains_key(&l) {
                    continue; // no delta for this layer
                }

                let offsets = meta.get("data_offsets")
                    .and_then(|v| v.as_array())
                    .map(|a| (a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize));
                let shape = meta.get("shape")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect::<Vec<_>>());
                let dtype = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("").to_string();

                if let (Some((ds, de)), Some(shape)) = (offsets, shape) {
                    gate_entries.push((l, key.clone(), TensorEntry {
                        dtype, shape,
                        data_start: ds,
                        data_end: de,
                    }));
                }
            }
        }

        if gate_entries.is_empty() {
            continue;
        }

        // Backup if needed
        if !premrg.exists() {
            std::fs::copy(file_path, &premrg)
                .with_context(|| format!("backup {}", file_path.display()))?;
            backup_created = true;
        }
        // Read original values from pristine source (read-only mmap)
        let source_file = std::fs::File::open(read_from)
            .with_context(|| format!("open source {}", read_from.display()))?;
        let source_data = unsafe { memmap2::Mmap::map(&source_file)
            .with_context(|| format!("mmap source {}", read_from.display()))? };
        // Open output file for in-place edits (mmap-mut, no copy needed)
        let out_file = std::fs::OpenOptions::new().read(true).write(true).open(file_path)
            .with_context(|| format!("open for write {}", file_path.display()))?;
        let mut data = unsafe { memmap2::MmapMut::map_mut(&out_file)
            .with_context(|| format!("mmap mut {}", file_path.display()))? };

        for (l, key, entry) in &gate_entries {
            let delta = &layer_deltas[l];
            let abs_start = data_start + entry.data_start;
            let abs_end = data_start + entry.data_end;

            // Dequantize the gate tensor
            let tensor_bytes = &source_data[abs_start..abs_end];

            // Need scales and biases from the same file
            let base_name = key.trim_end_matches(".weight");
            let scales_key = format!("{base_name}.scales");
            let biases_key = format!("{base_name}.biases");

            let scales_entry = header_json.get(&scales_key);
            let biases_entry = header_json.get(&biases_key);

            if scales_entry.is_none() || biases_entry.is_none() {
                debug!("Router L{l}: no scales/biases for {key}, skipping");
                continue;
            }

            let s_offsets = scales_entry.unwrap().get("data_offsets")
                .and_then(|v| v.as_array())
                .map(|a| (a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize))
                .unwrap();
            let b_offsets = biases_entry.unwrap().get("data_offsets")
                .and_then(|v| v.as_array())
                .map(|a| (a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize))
                .unwrap();

            let scales_bytes = &source_data[data_start + s_offsets.0..data_start + s_offsets.1];
            let biases_bytes = &source_data[data_start + b_offsets.0..data_start + b_offsets.1];

            // BF16 → f32
            let scales: Vec<f32> = scales_bytes.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let biases: Vec<f32> = biases_bytes.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();

            let rows = entry.shape[0]; // num_experts
            // For 3-bit: packed_cols * 32 / 3 is not integer.
            // Derive cols from scales count: n_scales_per_row * group_size.
            let n_groups_per_row = scales.len() / rows;
            let cols = n_groups_per_row * group_size;

            // Sanity: cols should match dim from function args
            if cols != dim {
                debug!("Router L{l}: cols={cols} != dim={dim}, skipping (wrong tensor?)");
                continue;
            }

            // Dequantize
            let mut weight_f32 = super::ane_weights::dequant_nbit(
                tensor_bytes, &scales, &biases, rows, cols, group_size, bits,
            );

            // Apply delta
            assert_eq!(weight_f32.len(), delta.len(),
                "Router L{l}: weight len {} != delta len {}", weight_f32.len(), delta.len());
            for i in 0..weight_f32.len() {
                weight_f32[i] += delta[i];
            }

            // Requantize
            let (new_data, new_scales, new_biases) = super::ane_weights::quantize_nbit(
                &weight_f32, rows, cols, group_size, bits,
            );

            // Write back weight data
            let orig_size = abs_end - abs_start;
            if new_data.len() != orig_size {
                tracing::warn!(
                    "Router L{l}: requantized size {} != original {}, skipping write-back",
                    new_data.len(), orig_size
                );
                tensors_merged += 1;
                continue;
            }
            data[abs_start..abs_end].copy_from_slice(&new_data);

            // Write back scales (f32 → bf16)
            let scales_bf16: Vec<u8> = new_scales.iter().flat_map(|&s| {
                let bits16 = (s.to_bits() >> 16) as u16;
                bits16.to_le_bytes().to_vec()
            }).collect();
            data[data_start + s_offsets.0..data_start + s_offsets.1]
                .copy_from_slice(&scales_bf16);

            // Write back biases (f32 → bf16)
            let biases_bf16: Vec<u8> = new_biases.iter().flat_map(|&b| {
                let bits16 = (b.to_bits() >> 16) as u16;
                bits16.to_le_bytes().to_vec()
            }).collect();
            data[data_start + b_offsets.0..data_start + b_offsets.1]
                .copy_from_slice(&biases_bf16);

            tensors_merged += 1;
            debug!("Router L{l}: merged delta into {key}");
        }

        // Flush mmap writes to disk
        data.flush()
            .with_context(|| format!("flush {}", file_path.display()))?;
        files_modified += 1;
    }

    let elapsed = start.elapsed().as_millis() as u64;
    info!("Router merge: {tensors_merged} tensors in {files_modified} files, {elapsed}ms");

    Ok(MergeReport {
        tensors_merged,
        files_modified,
        backup_created,
        elapsed_ms: elapsed,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip_4bit() {
        let rows = 8;
        let cols = 64;
        let group_size = 32;
        let bits = 4;

        // Random-ish values in typical weight range.
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.017).sin() * 0.5))
            .collect();

        let (data, scales, biases) = quantize_nbit(&values, rows, cols, group_size, bits);
        let recovered = dequant_nbit(&data, &scales, &biases, rows, cols, group_size, bits);

        let max_err = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // 4-bit with 15 levels: max error ≤ 1 quantization step.
        // step = range / 15, typical range ~1.0, so step ~0.067
        assert!(
            max_err < 0.1,
            "4-bit round-trip max error {max_err} too large"
        );
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_3bit() {
        let rows = 4;
        let cols = 96; // divisible by 32 (group_size) and allows 3-bit packing
        let group_size = 32;
        let bits = 3;

        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.031).cos() * 0.3))
            .collect();

        let (data, scales, biases) = quantize_nbit(&values, rows, cols, group_size, bits);
        let recovered = dequant_nbit(&data, &scales, &biases, rows, cols, group_size, bits);

        let max_err = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // 3-bit: 7 levels, coarser quantization.
        assert!(
            max_err < 0.15,
            "3-bit round-trip max error {max_err} too large"
        );
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_8bit() {
        let rows = 4;
        let cols = 64;
        let group_size = 64;
        let bits = 8;

        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.013).sin() * 2.0))
            .collect();

        let (data, scales, biases) = quantize_nbit(&values, rows, cols, group_size, bits);
        let recovered = dequant_nbit(&data, &scales, &biases, rows, cols, group_size, bits);

        let max_err = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // 8-bit: 255 levels, very fine.
        assert!(
            max_err < 0.02,
            "8-bit round-trip max error {max_err} too large"
        );
    }

    #[test]
    fn test_quantize_constant_group() {
        let rows = 1;
        let cols = 32;
        let group_size = 32;
        let bits = 4;

        let values = vec![0.42f32; rows * cols];
        let (data, scales, biases) = quantize_nbit(&values, rows, cols, group_size, bits);
        let recovered = dequant_nbit(&data, &scales, &biases, rows, cols, group_size, bits);

        // All values identical → scale=0, all qvals=0, bias=0.42
        assert_eq!(scales[0], 0.0);
        assert!((biases[0] - 0.42).abs() < 1e-6);
        for (a, b) in values.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "constant group: {a} vs {b}");
        }
        // All packed data should be zero.
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_compute_lora_delta() {
        // rank=2, d_in=3, d_out=2
        // A = [[1,0,0],[0,1,0]]  (2x3)
        // B = [[1,0],[0,1]]      (2x2)
        // scale = 1.0
        // delta = B @ A = [[1,0,0],[0,1,0]]  (2x3)
        let adapter = DeltaAdapter {
            a: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            b: vec![1.0, 0.0, 0.0, 1.0],
            rank: 2,
            d_in: 3,
            d_out: 2,
            scale: 1.0,
        };
        let delta = compute_lora_delta(&adapter);
        assert_eq!(delta, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_compute_lora_delta_with_scale() {
        let adapter = DeltaAdapter {
            a: vec![1.0, 2.0, 3.0, 4.0],
            b: vec![1.0, 0.0, 0.0, 1.0],
            rank: 2,
            d_in: 2,
            d_out: 2,
            scale: 0.5,
        };
        // scale=0.5, B@A = [[1*1+0*3, 1*2+0*4],[0*1+1*3, 0*2+1*4]] = [[1,2],[3,4]]
        // with scale: [[0.5,1.0],[1.5,2.0]]
        let delta = compute_lora_delta(&adapter);
        assert!((delta[0] - 0.5).abs() < 1e-6);
        assert!((delta[1] - 1.0).abs() < 1e-6);
        assert!((delta[2] - 1.5).abs() < 1e-6);
        assert!((delta[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = vec![0.0f32, 1.0, -1.0, 0.5, 0.001, 100.0];
        let bytes = f32_to_bf16_bytes(&values);
        let recovered = bf16_to_f32_slice(&bytes);
        for (a, b) in values.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            let tol = a.abs() * 0.01 + 1e-4; // ~1% relative + small absolute
            assert!(err < tol, "bf16 roundtrip: {a} vs {b}, err={err}");
        }
    }

    #[test]
    fn test_premrg_path() {
        let p = PathBuf::from("/models/model-00001-of-00004.safetensors");
        let premrg = premrg_path(&p);
        assert_eq!(
            premrg,
            PathBuf::from("/models/model-00001-of-00004.premrg.safetensors")
        );
    }

    #[test]
    fn test_merge_synthetic_safetensors() {
        // Build a minimal safetensors file with one quantized tensor.
        let rows = 4;
        let cols = 32;
        let group_size = 32;
        let bits = 4;

        let base_values: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.01) - 0.5)
            .collect();
        let (w_data, scales, biases) =
            quantize_nbit(&base_values, rows, cols, group_size, bits);
        let s_bytes = f32_to_bf16_bytes(&scales);
        let b_bytes = f32_to_bf16_bytes(&biases);

        let packed_shape_cols = cols * bits / 32;
        let st_buf = build_test_safetensors(
            "test.layers.0.self_attn.q_proj",
            &w_data,
            &s_bytes,
            &b_bytes,
            rows,
            packed_shape_cols,
            group_size,
        );

        // Parse it back.
        let (header_size, entries) = parse_safetensors_header(&st_buf).unwrap();
        assert!(entries.contains_key("test.layers.0.self_attn.q_proj.weight"));
        assert!(entries.contains_key("test.layers.0.self_attn.q_proj.scales"));
        assert!(entries.contains_key("test.layers.0.self_attn.q_proj.biases"));

        // Apply a simple delta (identity-like).
        let delta_val = 0.1f32;
        let mut delta_a = vec![0.0f32; 2 * cols]; // rank=2, d_in=cols
        let mut delta_b = vec![0.0f32; rows * 2]; // d_out=rows, rank=2
        // Set A[0,0]=1, B[0,0]=delta_val → adds delta_val to position [0,0]
        delta_a[0] = 1.0;
        delta_b[0] = delta_val;

        let pending = PendingDelta {
            base_name: "test.layers.0.self_attn.q_proj".to_string(),
            adapter: DeltaAdapter {
                a: delta_a,
                b: delta_b,
                rank: 2,
                d_in: cols,
                d_out: rows,
                scale: 1.0,
            },
        };

        let mut buf = st_buf;
        let data_start = 8 + header_size;
        apply_delta_to_buffer(&mut buf, data_start, &entries, &pending, bits, group_size)
            .unwrap();

        // Verify: dequantize the modified weight and check [0,0] shifted.
        let w_entry = &entries["test.layers.0.self_attn.q_proj.weight"];
        let s_entry = &entries["test.layers.0.self_attn.q_proj.scales"];
        let b_entry = &entries["test.layers.0.self_attn.q_proj.biases"];

        let new_w = &buf[data_start + w_entry.data_start..data_start + w_entry.data_end];
        let new_s = bf16_to_f32_slice(
            &buf[data_start + s_entry.data_start..data_start + s_entry.data_end],
        );
        let new_b = bf16_to_f32_slice(
            &buf[data_start + b_entry.data_start..data_start + b_entry.data_end],
        );
        let merged = dequant_nbit(new_w, &new_s, &new_b, rows, cols, group_size, bits);

        // Position [0,0]: should be approximately base_values[0] + delta_val
        let expected = base_values[0] + delta_val;
        let actual = merged[0];
        let err = (expected - actual).abs();
        assert!(
            err < 0.1,
            "merged[0,0]: expected ~{expected:.4}, got {actual:.4}, err={err:.4}"
        );

        // Other positions should be ~unchanged (within quantization noise).
        for i in 1..rows * cols {
            let err = (base_values[i] - merged[i]).abs();
            assert!(
                err < 0.1,
                "merged[{i}]: expected ~{:.4}, got {:.4}, err={err:.4}",
                base_values[i],
                merged[i]
            );
        }
    }

    #[test]
    fn test_merge_e2e_real_0_8b() {
        // Qwen3.5-0.8B-8bit: 24 layers, dim=1024, hidden=3584, bits=8, group_size=64
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-0.8B not found, skipping E2E merge test");
            return;
        }

        // Copy model to temp dir so we don't modify the real model.
        let tmp = tempfile::tempdir().unwrap();
        let tmp_dir = tmp.path();
        for entry in std::fs::read_dir(&model_dir).unwrap() {
            let e = entry.unwrap();
            let name = e.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".safetensors") || name_str == "config.json" {
                std::fs::copy(e.path(), tmp_dir.join(&name)).unwrap();
            }
        }

        // Snapshot original bytes for comparison.
        let orig_bytes = std::fs::read(tmp_dir.join("model.safetensors")).unwrap();

        // Build LoRA with one non-zero adapter: layer 0 down_proj [1024, 3584].
        // Linear attn indices = [0..23] minus [3,7,11,15,19,23].
        let linear_indices: Vec<usize> =
            (0..24).filter(|i| ![3, 7, 11, 15, 19, 23].contains(i)).collect();

        let n_layers = 24;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            use super::super::ane_lora::{LoraAdapter, LoraLayerAdapters};
            if i == 0 {
                // down_proj: d_out=1024, d_in=3584, rank=4
                let rank = 4usize;
                let d_in = 3584usize;
                let d_out = 1024usize;
                // A = small random, B = small constant so delta is visible
                let a: Vec<f32> = (0..rank * d_in)
                    .map(|j| (j as f32 * 0.0001).sin() * 0.1)
                    .collect();
                let mut b = vec![0.0f32; d_out * rank];
                // Set B[0,0] = 1.0 so row 0 of delta is non-zero
                b[0] = 1.0;
                layers.push(LoraLayerAdapters {
                    wq: None,
                    wv: None,
                    wo: None,
                    w2: Some(LoraAdapter { a, b, rank, d_in, d_out }),
                });
            } else {
                layers.push(LoraLayerAdapters {
                    wq: None, wv: None, wo: None, w2: None,
                });
            }
        }
        let lora = super::super::ane_lora::LoraModel {
            layers,
            config: super::super::ane_lora::LoraConfig {
                rank: 4,
                alpha: 4.0, // scale = alpha/rank = 1.0
                target_modules: vec!["w2".into()],
            },
        };

        // Run merge.
        let report = merge_lora_into_base(
            tmp_dir,
            &lora,
            "language_model.model",
            &linear_indices,
            8,  // bits
            64, // group_size
        )
        .expect("merge failed");

        eprintln!("Merge report: {report:?}");
        assert!(report.tensors_merged > 0, "should have merged tensors");
        assert!(report.files_modified > 0, "should have modified files");
        assert!(report.backup_created, "should have created backup");

        // Verify .premrg backup exists and matches original.
        let premrg_file = tmp_dir.join("model.premrg.safetensors");
        assert!(premrg_file.exists(), ".premrg backup should exist");
        let premrg_bytes = std::fs::read(&premrg_file).unwrap();
        assert_eq!(premrg_bytes, orig_bytes, ".premrg should match original");

        // Verify merged file differs from original.
        let merged_bytes = std::fs::read(tmp_dir.join("model.safetensors")).unwrap();
        assert_ne!(merged_bytes, orig_bytes, "merged should differ from original");

        // Read back merged down_proj and verify delta was applied.
        let (hdr_size, entries) = parse_safetensors_header(&merged_bytes).unwrap();
        let data_start = 8 + hdr_size;
        let base_name = "language_model.model.layers.0.mlp.down_proj";
        let w_e = &entries[&format!("{base_name}.weight")];
        let s_e = &entries[&format!("{base_name}.scales")];
        let b_e = &entries[&format!("{base_name}.biases")];

        // Dequantize merged weight.
        let rows = w_e.shape[0];
        let cols = w_e.shape[1] * 32 / 8;
        let merged_w = dequant_nbit(
            &merged_bytes[data_start + w_e.data_start..data_start + w_e.data_end],
            &bf16_to_f32_slice(&merged_bytes[data_start + s_e.data_start..data_start + s_e.data_end]),
            &bf16_to_f32_slice(&merged_bytes[data_start + b_e.data_start..data_start + b_e.data_end]),
            rows, cols, 64, 8,
        );

        // Dequantize original weight.
        let orig_w = dequant_nbit(
            &orig_bytes[data_start + w_e.data_start..data_start + w_e.data_end],
            &bf16_to_f32_slice(&orig_bytes[data_start + s_e.data_start..data_start + s_e.data_end]),
            &bf16_to_f32_slice(&orig_bytes[data_start + b_e.data_start..data_start + b_e.data_end]),
            rows, cols, 64, 8,
        );

        // Row 0 should have changed (B[0,0]=1.0 makes delta non-zero in row 0).
        let row0_diff: f32 = (0..cols)
            .map(|c| (merged_w[c] - orig_w[c]).abs())
            .sum();
        eprintln!("Row 0 total absolute diff: {row0_diff:.6}");
        assert!(row0_diff > 0.01, "row 0 should have visible delta, got {row0_diff}");

        // Row 1 should be ~unchanged (B[1,0]=0).
        let row1_diff: f32 = (0..cols)
            .map(|c| (merged_w[cols + c] - orig_w[cols + c]).abs())
            .sum();
        eprintln!("Row 1 total absolute diff: {row1_diff:.6}");
        // Allow small quantization noise but much less than row 0.
        assert!(
            row1_diff < row0_diff * 0.01,
            "row 1 should be ~unchanged: {row1_diff} vs row0 {row0_diff}"
        );

        eprintln!("E2E merge verified: delta applied to real 0.8B model weights");
    }

    #[test]
    fn test_merge_35b_moe_shared_expert_in_memory() {
        // Qwen3.5-35B-A3B: 4-bit, group_size=64, MoE with shared_expert.down_proj
        // Tests MoE fallback path + 4-bit quantization without copying 19GB to disk.
        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        if !model_dir.exists() {
            eprintln!("Qwen3.5-35B not found, skipping 35B merge test");
            return;
        }

        let bits = 4usize;
        let group_size = 64usize;
        // shared_expert.down_proj: d_out=2048 (dim), d_in=512 (moe_hidden)
        let d_out = 2048usize;
        let d_in = 512usize;
        let base_name = "language_model.model.layers.0.mlp.shared_expert.down_proj";

        // Read shard 1 (contains layer 0).
        let shard = model_dir.join("model-00001-of-00004.safetensors");
        let buf = std::fs::read(&shard).expect("read shard 1");
        let (header_size, entries) = parse_safetensors_header(&buf).unwrap();
        let data_start = 8 + header_size;

        // Verify tensor exists.
        let w_key = format!("{base_name}.weight");
        assert!(entries.contains_key(&w_key), "missing {w_key}");

        // Dequantize original weight.
        let w_e = &entries[&w_key];
        let s_e = &entries[&format!("{base_name}.scales")];
        let b_e = &entries[&format!("{base_name}.biases")];
        let rows = w_e.shape[0];
        let cols = w_e.shape[1] * 32 / bits;
        assert_eq!(rows, d_out);
        assert_eq!(cols, d_in);

        let orig_w = dequant_nbit(
            &buf[data_start + w_e.data_start..data_start + w_e.data_end],
            &bf16_to_f32_slice(&buf[data_start + s_e.data_start..data_start + s_e.data_end]),
            &bf16_to_f32_slice(&buf[data_start + b_e.data_start..data_start + b_e.data_end]),
            rows, cols, group_size, bits,
        );

        // Create LoRA delta: B[0,0]=1.0, A = small signal in first row.
        let rank = 4usize;
        let a: Vec<f32> = (0..rank * d_in).map(|j| if j < d_in { 0.05 } else { 0.0 }).collect();
        let mut b = vec![0.0f32; d_out * rank];
        b[0] = 1.0; // Only row 0 of delta is non-zero

        let pending = PendingDelta {
            base_name: base_name.to_string(),
            adapter: DeltaAdapter { a, b, rank, d_in, d_out, scale: 1.0 },
        };

        // Apply delta to a copy of the buffer.
        let mut buf_copy = buf.clone();
        apply_delta_to_buffer(&mut buf_copy, data_start, &entries, &pending, bits, group_size)
            .expect("apply delta failed");

        // Dequantize merged weight.
        let merged_w = dequant_nbit(
            &buf_copy[data_start + w_e.data_start..data_start + w_e.data_end],
            &bf16_to_f32_slice(&buf_copy[data_start + s_e.data_start..data_start + s_e.data_end]),
            &bf16_to_f32_slice(&buf_copy[data_start + b_e.data_start..data_start + b_e.data_end]),
            rows, cols, group_size, bits,
        );

        // Row 0: delta = B[0,0]*A[0,:] = 1.0 * [0.05, 0.05, ...] → shift of 0.05 per element.
        let row0_diff: f32 = (0..cols).map(|c| (merged_w[c] - orig_w[c]).abs()).sum();
        let row1_diff: f32 = (0..cols).map(|c| (merged_w[cols + c] - orig_w[cols + c]).abs()).sum();

        eprintln!("35B shared_expert.down_proj: {rows}x{cols}, 4-bit");
        eprintln!("Row 0 total diff: {row0_diff:.4} (expected ~{:.1})", 0.05 * cols as f32);
        eprintln!("Row 1 total diff: {row1_diff:.4}");

        assert!(row0_diff > 1.0, "row 0 should show delta: {row0_diff}");
        assert!(row1_diff < row0_diff * 0.05, "row 1 should be ~unchanged: {row1_diff} vs {row0_diff}");

        eprintln!("35B MoE 4-bit merge verified in-memory");
    }

    /// Build a minimal safetensors buffer with weight/scales/biases tensors.
    fn build_test_safetensors(
        base_name: &str,
        w_data: &[u8],
        s_bytes: &[u8],
        b_bytes: &[u8],
        rows: usize,
        packed_cols: usize,
        n_groups: usize,
    ) -> Vec<u8> {
        let w_start = 0usize;
        let w_end = w_data.len();
        let s_start = w_end;
        let s_end = s_start + s_bytes.len();
        let b_start = s_end;
        let b_end = b_start + b_bytes.len();

        let header = serde_json::json!({
            format!("{base_name}.weight"): {
                "dtype": "U32",
                "shape": [rows, packed_cols],
                "data_offsets": [w_start, w_end],
            },
            format!("{base_name}.scales"): {
                "dtype": "BF16",
                "shape": [rows, n_groups],
                "data_offsets": [s_start, s_end],
            },
            format!("{base_name}.biases"): {
                "dtype": "BF16",
                "shape": [rows, n_groups],
                "data_offsets": [b_start, b_end],
            },
        });

        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_size = header_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.extend_from_slice(w_data);
        buf.extend_from_slice(s_bytes);
        buf.extend_from_slice(b_bytes);
        buf
    }

    /// Merge REINFORCE router deltas from a binary file into model safetensors.
    ///
    /// The delta file is written by `routing_hook.py --eval`:
    ///   [n_layers: u32, ne: u32, dim: u32, then per layer: layer_idx: u32, delta: f32 × ne × dim]
    ///
    /// This test reads deltas, loads original router weights from safetensors,
    /// computes trained = original + delta, calls merge_router_into_safetensors.
    ///
    /// ```bash
    /// # Step 1: Generate deltas
    /// source .venv/bin/activate && python scripts/routing_hook.py --eval
    ///
    /// # Step 2: Merge into safetensors
    /// cargo test --features ane --release --lib -- "test_merge_router_from_deltas" --nocapture --ignored
    ///
    /// # Step 3: Re-eval (Python reloads from disk)
    /// python scripts/routing_hook.py --eval  # will use merged weights
    /// ```
    #[test]
    #[ignore]
    fn test_merge_router_from_deltas() {
        use std::io::Read;

        let home = std::env::var("HOME").unwrap();
        let deltas_path = format!("{home}/.nanobot/router_deltas.bin");

        if !std::path::Path::new(&deltas_path).exists() {
            eprintln!("SKIP: {deltas_path} not found. Run: python scripts/routing_hook.py --eval");
            return;
        }

        // Read deltas file
        let mut f = std::fs::File::open(&deltas_path).expect("open deltas");
        let mut header = [0u8; 12];
        f.read_exact(&mut header).expect("read header");
        let n_delta_layers = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let ne = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        eprintln!("Deltas: {n_delta_layers} layers, {ne} experts, dim={dim}");

        let mut layer_deltas: std::collections::HashMap<usize, Vec<f32>> =
            std::collections::HashMap::new();
        for _ in 0..n_delta_layers {
            let mut idx_buf = [0u8; 4];
            f.read_exact(&mut idx_buf).expect("read layer idx");
            let layer_idx = u32::from_le_bytes(idx_buf) as usize;

            let mut delta_buf = vec![0u8; ne * dim * 4];
            f.read_exact(&mut delta_buf).expect("read delta");
            let delta: Vec<f32> = delta_buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            layer_deltas.insert(layer_idx, delta);
        }
        eprintln!("Read {} layer deltas", layer_deltas.len());

        // Find model dir
        let model_dirs = [
            format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"),
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit"),
        ];
        let model_dir = model_dirs
            .iter()
            .find(|d| std::path::Path::new(d).exists())
            .expect("No 35B model found");
        let model_path = std::path::Path::new(model_dir);
        eprintln!("Model: {model_dir}");

        // Detect bits from config
        let config_str = std::fs::read_to_string(model_path.join("config.json"))
            .expect("read config");
        let config: serde_json::Value =
            serde_json::from_str(&config_str).expect("parse config");
        let quantization = config
            .get("quantization")
            .or_else(|| config.get("text_config").and_then(|tc| tc.get("quantization")));
        let bits = quantization
            .and_then(|q| q.get("bits").and_then(|b| b.as_u64()))
            .unwrap_or(4) as usize;
        let group_size = quantization
            .and_then(|q| q.get("group_size").and_then(|g| g.as_u64()))
            .unwrap_or(64) as usize;
        eprintln!("Quantization: {bits}-bit, group_size={group_size}");

        // Load original router weights by dequantizing from safetensors.
        // Use MmapTensorStore to read gate tensors directly.
        let n_layers = layer_deltas.keys().max().copied().unwrap_or(0) + 1;
        eprintln!("Building original + trained weight vectors for {n_layers} layers...");

        // Build original and trained weight vectors
        // For layers WITHOUT deltas: empty vec (skipped by merge)
        // For layers WITH deltas: load original from safetensors, compute trained = orig + delta
        let mut original_weights: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let mut trained_weights: Vec<Vec<f32>> = vec![Vec::new(); n_layers];

        // Read original gate weights from safetensors (mmap for >2GB files)
        let st_files = list_safetensors(model_path).expect("list safetensors");
        for file_path in &st_files {
            let file = std::fs::File::open(file_path).expect("open safetensors");
            let raw = unsafe { memmap2::Mmap::map(&file).expect("mmap safetensors") };
            if raw.len() < 8 { continue; }
            let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
            let header_json: serde_json::Value =
                serde_json::from_slice(&raw[8..8 + header_len]).expect("parse header");
            let data_start = 8 + header_len;

            if let Some(obj) = header_json.as_object() {
                for (key, meta) in obj {
                    if key == "__metadata__" { continue; }
                    // Match "mlp.gate.weight" exactly — NOT "mlp.gate_proj.weight" (expert FFN)
                    if !key.ends_with("mlp.gate.weight") { continue; }

                    let layer_idx = key.split("layers.")
                        .nth(1)
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse::<usize>().ok());
                    let Some(l) = layer_idx else { continue };
                    if !layer_deltas.contains_key(&l) { continue; }

                    let offsets = meta.get("data_offsets")
                        .and_then(|v| v.as_array())
                        .map(|a| (a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize));
                    let shape = meta.get("shape")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect::<Vec<_>>());

                    let Some((ds, de)) = offsets else { continue };
                    let Some(shape) = shape else { continue };
                    let rows = shape[0];
                    // For 3-bit: packed_cols * 32 / 3 is not integer.
                    // Use scales shape to get actual n_groups, then cols = n_groups * group_size.
                    let s_shape = obj.get(&format!("{}.scales", key.trim_end_matches(".weight")))
                        .and_then(|m| m.get("shape"))
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect::<Vec<_>>());
                    let n_groups = s_shape.as_ref().map(|s| s[1]).unwrap_or(dim / group_size);
                    let cols = n_groups * group_size;

                    let w_bytes = &raw[data_start + ds..data_start + de];

                    // Find scales and biases
                    let base_name = key.trim_end_matches(".weight");
                    let s_key = format!("{base_name}.scales");
                    let b_key = format!("{base_name}.biases");
                    let s_meta = obj.get(&s_key);
                    let b_meta = obj.get(&b_key);
                    if s_meta.is_none() || b_meta.is_none() { continue; }

                    let s_off = s_meta.unwrap().get("data_offsets").unwrap().as_array().unwrap();
                    let b_off = b_meta.unwrap().get("data_offsets").unwrap().as_array().unwrap();
                    let s_bytes = &raw[data_start + s_off[0].as_u64().unwrap() as usize
                        ..data_start + s_off[1].as_u64().unwrap() as usize];
                    let b_bytes = &raw[data_start + b_off[0].as_u64().unwrap() as usize
                        ..data_start + b_off[1].as_u64().unwrap() as usize];

                    let scales: Vec<f32> = s_bytes.chunks_exact(2)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect();
                    let biases: Vec<f32> = b_bytes.chunks_exact(2)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect();

                    let orig = super::super::ane_weights::dequant_nbit(
                        w_bytes, &scales, &biases, rows, cols, group_size, bits,
                    );

                    let delta = &layer_deltas[&l];
                    let trained: Vec<f32> = orig.iter().zip(delta.iter())
                        .map(|(o, d)| o + d).collect();

                    original_weights[l] = orig;
                    trained_weights[l] = trained;
                    eprintln!("  L{l}: loaded {rows}×{cols} gate, delta max={:.6}",
                        delta.iter().map(|d| d.abs()).fold(0.0f32, f32::max));
                }
            }
        }

        let layers_loaded = original_weights.iter().filter(|w| !w.is_empty()).count();
        eprintln!("Loaded {layers_loaded} original router gates");

        if layers_loaded == 0 {
            eprintln!("SKIP: No router gates found in safetensors");
            return;
        }

        // Merge!
        eprintln!("\nMerging router deltas into safetensors...");
        let t0 = std::time::Instant::now();
        let report = merge_router_into_safetensors(
            model_path,
            &trained_weights,
            &original_weights,
            ne,
            dim,
            bits,
            group_size,
        )
        .expect("merge failed");

        eprintln!(
            "  Merged {} tensors in {} files, {:.1}s",
            report.tensors_merged,
            report.files_modified,
            t0.elapsed().as_secs_f64()
        );
        eprintln!("  Backup: {}", if report.backup_created { "created" } else { "reused existing" });
        eprintln!("\nDone! Re-run eval to measure effect:");
        eprintln!("  python scripts/routing_hook.py --eval");
    }
}

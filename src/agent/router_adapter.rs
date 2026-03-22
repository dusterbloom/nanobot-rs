//! Router adapter sidecar — fp32 MoE gate weights saved separately from
//! the quantized base model. Avoids the lossy dequant→requant roundtrip
//! that destroys 3-bit models (5% baseline loss from quantization noise).
//!
//! The adapter is a simple binary file: `router_adapter.bin` next to the
//! model safetensors. Format:
//!   [n_layers: u32, num_experts: u32, dim: u32]
//!   For each layer with trained gates:
//!     [layer_idx: u32, gate_weights: f32 × num_experts × dim]
//!   Layers without trained gates are skipped (use quantized base).

use std::io::{self, Read, Write};
use std::path::Path;

const ADAPTER_FILENAME: &str = "router_adapter.bin";

/// Loaded router adapter with per-layer fp32 gate weights.
#[derive(Debug, Clone)]
pub struct RouterAdapter {
    gates: Vec<Option<Vec<f32>>>,
    pub num_experts: usize,
    pub dim: usize,
}

impl RouterAdapter {
    /// Save trained router gates to a sidecar file.
    ///
    /// `gates[l]` is non-empty for MoE layers with trained weights,
    /// empty for layers without MoE or without training.
    pub fn save(model_dir: &Path, gates: &[Vec<f32>], num_experts: usize, dim: usize) -> io::Result<()> {
        let path = model_dir.join(ADAPTER_FILENAME);

        let trained_layers: Vec<(usize, &Vec<f32>)> = gates
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .collect();

        let mut f = std::fs::File::create(&path)?;

        // Header
        f.write_all(&(trained_layers.len() as u32).to_le_bytes())?;
        f.write_all(&(num_experts as u32).to_le_bytes())?;
        f.write_all(&(dim as u32).to_le_bytes())?;

        // Per-layer gates
        for (layer_idx, gate) in &trained_layers {
            f.write_all(&(*layer_idx as u32).to_le_bytes())?;
            let bytes: Vec<u8> = gate.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bytes)?;
        }

        tracing::info!(
            "Router adapter saved: {} layers to {}",
            trained_layers.len(),
            path.display()
        );
        Ok(())
    }

    /// Load router adapter from sidecar file. Returns None if file doesn't exist.
    pub fn load(model_dir: &Path) -> io::Result<Option<Self>> {
        let path = model_dir.join(ADAPTER_FILENAME);
        if !path.exists() {
            return Ok(None);
        }

        let mut f = std::fs::File::open(&path)?;

        // Header
        let mut header = [0u8; 12];
        f.read_exact(&mut header)?;
        let n_trained = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let num_experts = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;

        // Find max layer index to size the vec
        let gate_size = num_experts * dim;
        let mut max_layer = 0usize;
        let mut layer_data: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n_trained);

        for _ in 0..n_trained {
            let mut idx_buf = [0u8; 4];
            f.read_exact(&mut idx_buf)?;
            let layer_idx = u32::from_le_bytes(idx_buf) as usize;
            max_layer = max_layer.max(layer_idx);

            let mut gate_bytes = vec![0u8; gate_size * 4];
            f.read_exact(&mut gate_bytes)?;
            let gate: Vec<f32> = gate_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            layer_data.push((layer_idx, gate));
        }

        let mut gates = vec![None; max_layer + 1];
        for (idx, gate) in layer_data {
            gates[idx] = Some(gate);
        }

        tracing::info!(
            "Router adapter loaded: {n_trained} layers from {}",
            path.display()
        );
        Ok(Some(Self {
            gates,
            num_experts,
            dim,
        }))
    }

    /// Get fp32 gate weights for a layer. Returns None if this layer
    /// should use the quantized base gate.
    pub fn gate_for_layer(&self, layer: usize) -> Option<&[f32]> {
        self.gates.get(layer).and_then(|g| g.as_deref())
    }

    /// Number of layers with trained gates.
    pub fn n_trained(&self) -> usize {
        self.gates.iter().filter(|g| g.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_adapter_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let ne = 32;
        let dim = 64;

        // Create gates for layers 0, 3, 7 (skip 1, 2, 4, 5, 6)
        let mut gates = vec![Vec::new(); 8];
        gates[0] = (0..ne * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        gates[3] = (0..ne * dim).map(|i| (i as f32 * 0.02).cos()).collect();
        gates[7] = (0..ne * dim).map(|i| (i as f32 * 0.03).sin()).collect();

        // Save
        RouterAdapter::save(dir.path(), &gates, ne, dim).unwrap();
        assert!(dir.path().join(ADAPTER_FILENAME).exists());

        // Load
        let adapter = RouterAdapter::load(dir.path()).unwrap().unwrap();
        assert_eq!(adapter.num_experts, ne);
        assert_eq!(adapter.dim, dim);
        assert_eq!(adapter.n_trained(), 3);

        // Verify gate values
        assert!(adapter.gate_for_layer(0).is_some());
        assert!(adapter.gate_for_layer(1).is_none()); // not trained
        assert!(adapter.gate_for_layer(3).is_some());
        assert!(adapter.gate_for_layer(7).is_some());

        let g0 = adapter.gate_for_layer(0).unwrap();
        assert_eq!(g0.len(), ne * dim);
        for i in 0..ne * dim {
            assert!(
                (g0[i] - (i as f32 * 0.01).sin()).abs() < 1e-6,
                "mismatch at {i}: {} vs {}",
                g0[i],
                (i as f32 * 0.01).sin()
            );
        }
    }

    #[test]
    fn test_router_adapter_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let result = RouterAdapter::load(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_router_adapter_file_size() {
        let dir = tempfile::tempdir().unwrap();
        let ne = 256;
        let dim = 2048;

        // 40 layers of trained gates
        let gates: Vec<Vec<f32>> = (0..40)
            .map(|l| (0..ne * dim).map(|i| ((l * ne * dim + i) as f32 * 0.001).sin()).collect())
            .collect();

        RouterAdapter::save(dir.path(), &gates, ne, dim).unwrap();

        let size = std::fs::metadata(dir.path().join(ADAPTER_FILENAME))
            .unwrap()
            .len();
        // Header: 12 bytes + 40 * (4 + 256 * 2048 * 4) = 12 + 40 * 2097156 = ~80 MB
        let expected = 12 + 40 * (4 + ne * dim * 4);
        assert_eq!(size as usize, expected);

        // Should be ~80 MB — fits easily in memory
        let mb = size as f64 / 1e6;
        assert!(mb < 100.0, "adapter should be < 100 MB, got {mb:.1} MB");
    }
}

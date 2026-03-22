// Factored vocabulary projection — two-stage lm_head to cut bandwidth 340x.
//
// Instead of scanning the full embedding matrix (248K × 2560 = 1.2 GB) per token,
// we cluster embedding rows offline (k-means), then at inference:
//   Stage 1: small matmul against cluster centroids → pick top-k clusters
//   Stage 2: partial matmul against only selected cluster rows
//
// With 1000 clusters, top-3: ~3.5 MB read vs 1.2 GB = 340x bandwidth reduction.

use std::io::{Read, Write};
use std::path::Path;

/// Offline-computed cluster assignment for the vocabulary embedding matrix.
#[derive(Debug, Clone)]
pub struct VocabClusters {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Centroid matrix, flat `[n_clusters * dim]`. Row-major.
    pub centroids: Vec<f32>,
    /// Cluster assignment per token: `cluster_of[token_id] = cluster_index`.
    pub cluster_of: Vec<u16>,
    /// Token IDs sorted by cluster.
    pub cluster_members: Vec<u32>,
    /// Range boundaries: cluster `c` spans `cluster_members[cluster_ranges[c]..cluster_ranges[c+1]]`.
    pub cluster_ranges: Vec<usize>,
    /// Hidden dimension.
    pub dim: usize,
}

/// Result of factored projection — sparse logits.
pub struct FactoredLogits {
    /// Full-vocab logits vector. Tokens not in selected clusters get `NEG_INFINITY`.
    pub logits: Vec<f32>,
    /// Number of embedding rows actually read (for bandwidth accounting).
    pub rows_read: usize,
}

/// Full dense projection (baseline for comparison).
pub fn full_project(h: &[f32], weights: &[f32], vocab_size: usize, dim: usize) -> Vec<f32> {
    super::ane_forward::cpu_matmul(weights, h, vocab_size, dim, 1)
}

/// Cluster embedding rows using k-means.
///
/// `weights` is `[vocab_size * dim]` row-major. Returns cluster assignments
/// and centroids. Uses k-means++ initialization and `max_iter` iterations.
pub fn cluster_embeddings(
    weights: &[f32],
    vocab_size: usize,
    dim: usize,
    n_clusters: usize,
    max_iter: usize,
) -> VocabClusters {
    assert_eq!(weights.len(), vocab_size * dim);
    assert!(n_clusters > 0 && n_clusters <= vocab_size);

    // ── K-means++ initialization ──
    let mut centroids = vec![0.0f32; n_clusters * dim];
    let mut rng_state: u64 = 42;
    let mut next_rng = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // First centroid: random row
    let first = (next_rng() as usize) % vocab_size;
    centroids[..dim].copy_from_slice(&weights[first * dim..(first + 1) * dim]);

    // Remaining centroids: proportional to squared distance
    let mut dists = vec![f32::MAX; vocab_size];
    for c in 1..n_clusters {
        // Update distances to nearest centroid
        for i in 0..vocab_size {
            let row = &weights[i * dim..(i + 1) * dim];
            let prev = &centroids[(c - 1) * dim..c * dim];
            let d: f32 = row.iter().zip(prev).map(|(a, b)| (a - b) * (a - b)).sum();
            dists[i] = dists[i].min(d);
        }
        // Weighted random selection
        let total: f64 = dists.iter().map(|&d| d as f64).sum();
        let threshold = (next_rng() as f64 / u64::MAX as f64) * total;
        let mut cumulative = 0.0f64;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            cumulative += d as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids[c * dim..(c + 1) * dim]
            .copy_from_slice(&weights[chosen * dim..(chosen + 1) * dim]);
    }

    // ── K-means iterations ──
    let mut assignments = vec![0u16; vocab_size];

    for _iter in 0..max_iter {
        // Assign each row to nearest centroid
        for i in 0..vocab_size {
            let row = &weights[i * dim..(i + 1) * dim];
            let mut best_c = 0u16;
            let mut best_d = f32::MAX;
            for c in 0..n_clusters {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let d: f32 = row.iter().zip(cent).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d {
                    best_d = d;
                    best_c = c as u16;
                }
            }
            assignments[i] = best_c;
        }

        // Recompute centroids
        let mut counts = vec![0u32; n_clusters];
        centroids.fill(0.0);
        for i in 0..vocab_size {
            let c = assignments[i] as usize;
            counts[c] += 1;
            let row = &weights[i * dim..(i + 1) * dim];
            for d in 0..dim {
                centroids[c * dim + d] += row[d];
            }
        }
        for c in 0..n_clusters {
            if counts[c] > 0 {
                let scale = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] *= scale;
                }
            }
        }
    }

    // ── Build sorted member lists ──
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];
    for (i, &c) in assignments.iter().enumerate() {
        buckets[c as usize].push(i as u32);
    }

    let mut cluster_members = Vec::with_capacity(vocab_size);
    let mut cluster_ranges = Vec::with_capacity(n_clusters + 1);
    cluster_ranges.push(0);
    for bucket in &buckets {
        cluster_members.extend_from_slice(bucket);
        cluster_ranges.push(cluster_members.len());
    }

    VocabClusters {
        n_clusters,
        centroids,
        cluster_of: assignments,
        cluster_members,
        cluster_ranges,
        dim,
    }
}

/// Two-stage factored projection.
///
/// Stage 1: `centroids[n_clusters, dim] @ h[dim]` → pick `top_k` clusters.
/// Stage 2: For each selected cluster, compute logits only for member tokens.
/// Non-selected tokens get `NEG_INFINITY`.
pub fn factored_project(
    h: &[f32],
    weights: &[f32],
    clusters: &VocabClusters,
    top_k: usize,
) -> FactoredLogits {
    let dim = clusters.dim;
    let vocab_size = clusters.cluster_of.len();
    let n_clusters = clusters.n_clusters;

    // Stage 1: cluster logits
    let mut cluster_logits = vec![0.0f32; n_clusters];
    for c in 0..n_clusters {
        let cent = &clusters.centroids[c * dim..(c + 1) * dim];
        cluster_logits[c] = cent.iter().zip(h).map(|(a, b)| a * b).sum();
    }

    // Pick top-k clusters
    let k = top_k.min(n_clusters);
    let mut indices: Vec<usize> = (0..n_clusters).collect();
    indices.sort_unstable_by(|&a, &b| {
        cluster_logits[b]
            .partial_cmp(&cluster_logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = &indices[..k];

    // Stage 2: compute logits for selected cluster members only
    let mut logits = vec![f32::NEG_INFINITY; vocab_size];
    let mut rows_read = 0usize;

    for &c in selected {
        let start = clusters.cluster_ranges[c];
        let end = clusters.cluster_ranges[c + 1];
        for &token_id in &clusters.cluster_members[start..end] {
            let row = &weights[token_id as usize * dim..(token_id as usize + 1) * dim];
            logits[token_id as usize] = row.iter().zip(h).map(|(a, b)| a * b).sum();
            rows_read += 1;
        }
    }

    FactoredLogits { logits, rows_read }
}

impl VocabClusters {
    /// Save to binary file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        let header = [
            self.n_clusters as u32,
            self.dim as u32,
            self.cluster_of.len() as u32, // vocab_size
            0, // reserved
        ];
        for &v in &header {
            f.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.centroids {
            f.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.cluster_of {
            f.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.cluster_members {
            f.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.cluster_ranges {
            f.write_all(&(v as u32).to_le_bytes())?;
        }
        Ok(())
    }

    /// Load from binary file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut pos = 0usize;

        let read_u32 = |pos: &mut usize| -> u32 {
            let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v
        };

        let n_clusters = read_u32(&mut pos) as usize;
        let dim = read_u32(&mut pos) as usize;
        let vocab_size = read_u32(&mut pos) as usize;
        let _reserved = read_u32(&mut pos);

        let mut centroids = vec![0.0f32; n_clusters * dim];
        for v in &mut centroids {
            *v = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
        }

        let mut cluster_of = vec![0u16; vocab_size];
        for v in &mut cluster_of {
            *v = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            pos += 2;
        }

        let mut cluster_members = vec![0u32; vocab_size];
        for v in &mut cluster_members {
            *v = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
        }

        let mut cluster_ranges = vec![0usize; n_clusters + 1];
        for v in &mut cluster_ranges {
            *v = read_u32(&mut pos) as usize;
        }

        Ok(VocabClusters {
            n_clusters,
            centroids,
            cluster_of,
            cluster_members,
            cluster_ranges,
            dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic embedding with clear cluster structure.
    /// 8 clusters of 32 tokens each = 256 tokens, dim=64.
    /// Each cluster has a distinct centroid with gaussian noise per row.
    fn make_clustered_embeddings() -> (Vec<f32>, usize, usize, usize) {
        let n_clusters = 8;
        let tokens_per_cluster = 32;
        let vocab = n_clusters * tokens_per_cluster; // 256
        let dim = 64;

        let mut weights = vec![0.0f32; vocab * dim];
        let mut rng: u64 = 123;
        let mut randf = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Map to approximately [-1, 1]
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        for c in 0..n_clusters {
            // Cluster centroid: distinct direction
            let mut centroid = vec![0.0f32; dim];
            for d in 0..dim {
                centroid[d] = randf() * 5.0; // large signal
            }

            for t in 0..tokens_per_cluster {
                let token_id = c * tokens_per_cluster + t;
                for d in 0..dim {
                    weights[token_id * dim + d] = centroid[d] + randf() * 0.3; // small noise
                }
            }
        }

        (weights, vocab, dim, n_clusters)
    }

    /// Make a hidden state that strongly activates one cluster.
    fn make_hidden_state(weights: &[f32], target_token: usize, dim: usize) -> Vec<f32> {
        // Use the target token's embedding as the hidden state (guarantees it's the argmax)
        weights[target_token * dim..(target_token + 1) * dim].to_vec()
    }

    #[test]
    fn test_factored_top1_matches_full() {
        let (weights, vocab, dim, n_clusters) = make_clustered_embeddings();
        let clusters = cluster_embeddings(&weights, vocab, dim, n_clusters, 20);

        // Test with multiple hidden states targeting different clusters
        for target_token in [0, 31, 64, 128, 200, 255] {
            let h = make_hidden_state(&weights, target_token, dim);

            let full_logits = full_project(&h, &weights, vocab, dim);
            let full_top1 = full_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            let factored = factored_project(&h, &weights, &clusters, 3);
            let fact_top1 = factored
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            assert_eq!(
                full_top1, fact_top1,
                "Top-1 mismatch for target_token={target_token}: full={full_top1}, factored={fact_top1}"
            );
        }
    }

    #[test]
    fn test_factored_top5_recall() {
        let (weights, vocab, dim, n_clusters) = make_clustered_embeddings();
        let clusters = cluster_embeddings(&weights, vocab, dim, n_clusters, 20);

        let h = make_hidden_state(&weights, 100, dim);

        let full_logits = full_project(&h, &weights, vocab, dim);
        let mut full_ranked: Vec<(usize, f32)> =
            full_logits.iter().copied().enumerate().collect();
        full_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let full_top5: Vec<usize> = full_ranked.iter().take(5).map(|&(i, _)| i).collect();

        let factored = factored_project(&h, &weights, &clusters, 3);

        for &tok in &full_top5 {
            assert!(
                factored.logits[tok] > f32::NEG_INFINITY,
                "Top-5 token {tok} is missing (NEG_INFINITY) in factored output"
            );
        }
    }

    #[test]
    fn test_factored_bandwidth_reduction() {
        let (weights, vocab, dim, n_clusters) = make_clustered_embeddings();
        let clusters = cluster_embeddings(&weights, vocab, dim, n_clusters, 20);

        let h = make_hidden_state(&weights, 50, dim);
        let factored = factored_project(&h, &weights, &clusters, 3);

        // Stage 1 reads centroids: n_clusters * dim = 8 * 64 = 512 floats
        // Stage 2 reads selected rows: rows_read * dim
        let total_reads = n_clusters * dim + factored.rows_read * dim;
        let full_reads = vocab * dim;

        let ratio = total_reads as f64 / full_reads as f64;
        assert!(
            ratio < 0.50,
            "Bandwidth reduction too small: {:.1}% of full (expected < 50%)",
            ratio * 100.0
        );
    }

    #[test]
    fn test_cluster_save_load_roundtrip() {
        let (weights, vocab, dim, n_clusters) = make_clustered_embeddings();
        let clusters = cluster_embeddings(&weights, vocab, dim, n_clusters, 20);

        let tmp = std::env::temp_dir().join("test_vocab_clusters.bin");
        clusters.save(&tmp).unwrap();
        let loaded = VocabClusters::load(&tmp).unwrap();

        assert_eq!(clusters.n_clusters, loaded.n_clusters);
        assert_eq!(clusters.dim, loaded.dim);
        assert_eq!(clusters.centroids, loaded.centroids);
        assert_eq!(clusters.cluster_of, loaded.cluster_of);
        assert_eq!(clusters.cluster_members, loaded.cluster_members);
        assert_eq!(clusters.cluster_ranges, loaded.cluster_ranges);

        std::fs::remove_file(&tmp).ok();
    }
}

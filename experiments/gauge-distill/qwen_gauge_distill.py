#!/usr/bin/env python3
"""
Gauge Theory Distillation — gl(n,ℝ) Lie-algebra fiber bundle distillation.

Distills a teacher LLM into a tiny dynamics core by replacing all transformer
blocks with a single FiberBundleLayer that transports hidden states through
a gl(n,ℝ) gauge connection. The frozen embedding, final norm, and LM head
are reused from the teacher — only the fiber is trained.

Designed for the nanobot-rs project: exports ANE-compatible traced models
using conv1x1 ops (matching the ANE BLOBFILE pipeline in ane_weights.rs).

Usage:
    source .venv/bin/activate
    python experiments/gauge-distill/qwen_gauge_distill.py
    python experiments/gauge-distill/qwen_gauge_distill.py --algebra gl6
    python experiments/gauge-distill/qwen_gauge_distill.py --model Qwen/Qwen2.5-3B-Instruct
    python experiments/gauge-distill/qwen_gauge_distill.py --local-model ~/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Gauge Theory Distillation — gl(n,R) fiber bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", type=str, default=None,
                   help="HuggingFace model ID (auto-detect if not set)")
    p.add_argument("--local-model", type=str, default=None,
                   help="Path to local MLX-format model directory")
    p.add_argument("--algebra", type=str, default="gl4",
                   choices=["gl4", "gl6", "gl8"],
                   help="Lie algebra (default: gl4)")
    p.add_argument("--steps", type=int, default=None,
                   help="Distillation steps (auto by model size)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (auto by model size)")
    p.add_argument("--device", type=str, default=None,
                   help="cpu | cuda | mps (auto-detect)")
    p.add_argument("--gen-tokens", type=int, default=60,
                   help="Tokens to generate (default: 60)")
    p.add_argument("--export-ane", action="store_true",
                   help="Run ANE export after distillation")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile (requires PyTorch 2.0+)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--seq-len", type=int, default=64,
                   help="Sequence length for training (default: 64)")
    p.add_argument("--skip-distill", action="store_true",
                   help="Skip distillation, only run algebra checks")
    return p.parse_args()


def detect_device(requested=None):
    """Auto-detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — gl(n,ℝ) LIE ALGEBRA
# ═══════════════════════════════════════════════════════════════

def algebra_dims(name):
    """Return (n, fiber_dim, visible_dim, dark_dim) for a named algebra."""
    table = {
        "gl4": (4, 16, 6, 10),
        "gl6": (6, 36, 15, 21),
        "gl8": (8, 64, 28, 36),
    }
    return table[name]


def build_basis(n):
    """Build the standard basis E_{ij} of gl(n,ℝ).

    Returns tensor of shape (n², n, n) where basis[n*i+j] = E_{ij}.
    """
    basis = torch.zeros(n * n, n, n)
    for i in range(n):
        for j in range(n):
            basis[n * i + j, i, j] = 1.0
    return basis


def lie_bracket(a, b):
    """Compute [A, B] = AB - BA."""
    return a @ b - b @ a


def build_projectors(n):
    """Build visible (so(n)) and dark (sym(n)) projectors for gl(n,ℝ).

    Returns P_vis, P_dark as (n², n²) matrices operating on the
    vectorized (flattened) representation of gl(n) elements.
    """
    fiber_dim = n * n
    vis_dim = n * (n - 1) // 2
    dark_dim = n * (n + 1) // 2

    # Build orthonormal bases in vectorized form
    V_vis = torch.zeros(fiber_dim, vis_dim)
    V_dark = torch.zeros(fiber_dim, dark_dim)

    # Antisymmetric basis: (E_{ij} - E_{ji}) / sqrt(2) for i < j
    col = 0
    for i in range(n):
        for j in range(i + 1, n):
            vec = torch.zeros(fiber_dim)
            vec[n * i + j] = 1.0 / math.sqrt(2)
            vec[n * j + i] = -1.0 / math.sqrt(2)
            V_vis[:, col] = vec
            col += 1

    # Symmetric basis: (E_{ij} + E_{ji}) / sqrt(2) for i < j, plus E_{ii}
    col = 0
    for i in range(n):
        for j in range(i + 1, n):
            vec = torch.zeros(fiber_dim)
            vec[n * i + j] = 1.0 / math.sqrt(2)
            vec[n * j + i] = 1.0 / math.sqrt(2)
            V_dark[:, col] = vec
            col += 1
    for i in range(n):
        vec = torch.zeros(fiber_dim)
        vec[n * i + i] = 1.0
        V_dark[:, col] = vec
        col += 1

    P_vis = V_vis @ V_vis.T
    P_dark = V_dark @ V_dark.T
    return P_vis, P_dark


def verify_lie_algebra(n):
    """Run all Lie algebra verification checks."""
    print("\n── Lie Algebra Verification ──")
    basis = build_basis(n)
    fiber_dim = n * n
    print(f"Basis: {fiber_dim} elements of shape ({n}, {n})")

    # Bracket closure: [E_a, E_b] must be expressible in terms of basis
    max_residual = 0.0
    for a in range(fiber_dim):
        for b in range(fiber_dim):
            bracket = lie_bracket(basis[a], basis[b])
            # Reconstruct from basis
            coeffs = bracket.reshape(-1)  # Already in basis coords since E_{ij} are standard
            reconstructed = torch.zeros(n, n)
            for k in range(fiber_dim):
                reconstructed += coeffs[k] * basis[k]
            residual = (bracket - reconstructed).abs().max().item()
            max_residual = max(max_residual, residual)
    status = "PASS" if max_residual < 1e-5 else "FAIL"
    print(f"Bracket closure: {max_residual:.1e} {status}")

    # Jacobi identity: [A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0
    torch.manual_seed(0)
    max_jacobi = 0.0
    for _ in range(200):
        A = torch.randn(n, n)
        B = torch.randn(n, n)
        C = torch.randn(n, n)
        jacobi = (lie_bracket(A, lie_bracket(B, C))
                   + lie_bracket(B, lie_bracket(C, A))
                   + lie_bracket(C, lie_bracket(A, B)))
        max_jacobi = max(max_jacobi, jacobi.abs().max().item())
    status = "PASS" if max_jacobi < 1e-4 else "FAIL"
    print(f"Jacobi identity: {max_jacobi:.1e} {status}")

    # Projector checks
    P_vis, P_dark = build_projectors(n)

    vis_rank = torch.linalg.matrix_rank(P_vis).item()
    dark_rank = torch.linalg.matrix_rank(P_dark).item()
    expected_vis = n * (n - 1) // 2
    expected_dark = n * (n + 1) // 2
    print(f"P_vis rank: {vis_rank} (expected {expected_vis}) "
          f"{'PASS' if vis_rank == expected_vis else 'FAIL'}")
    print(f"P_dark rank: {dark_rank} (expected {expected_dark}) "
          f"{'PASS' if dark_rank == expected_dark else 'FAIL'}")

    ortho = (P_vis @ P_dark).norm().item()
    print(f"Orthogonality ||P_vis @ P_dark||: {ortho:.1e} "
          f"{'PASS' if ortho < 1e-10 else 'FAIL'}")

    complete = (P_vis + P_dark - torch.eye(fiber_dim)).norm().item()
    print(f"Completeness ||P_vis + P_dark - I||: {complete:.1e} "
          f"{'PASS' if complete < 1e-6 else 'FAIL'}")

    return P_vis, P_dark


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DARK-MODE SCALING LAW
# ═══════════════════════════════════════════════════════════════

def dark_scaling_law(n_dark):
    """errors(N) = 209 * 0.881^N, accuracy = 100 - errors/100."""
    errors = 209.0 * (0.881 ** n_dark)
    return max(0.0, 100.0 - errors)


def print_scaling_table():
    """Print the dark-mode scaling law table."""
    print("\n── Dark-Mode Scaling Law ──")
    print("Formula: errors(N) = 209 * 0.881^N")
    print(f"{'N':>4}  {'errors':>8}  {'accuracy':>8}")
    print("─" * 24)
    for n in [0, 1, 3, 5, 10, 15, 21, 30, 36]:
        errors = 209.0 * (0.881 ** n)
        acc = max(0.0, 100.0 - errors)
        print(f"{n:>4}  {errors:>8.2f}  {acc:>7.2f}%")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — MODEL LOADING
# ═══════════════════════════════════════════════════════════════

# Model chain: start small for fast iteration, scale up once validated
# 0.8B → 2B → 35B-A3B (MoE, 3B active, hidden=2560 — ideal for ANE)
MODEL_CHAIN = [
    "Qwen/Qwen2.5-0.5B-Instruct",   # ~1GB, runs on CPU in seconds
    "gpt2",                            # last resort fallback
]


def get_model_parts(teacher):
    """Extract frozen components regardless of model family.

    Returns dict with: embed, norm, lm_head, hidden_dim, num_layers,
    vocab_size, has_pos_embed, family, and optionally pos_embed.
    """
    cfg = teacher.config
    if hasattr(cfg, "hidden_size"):
        # Qwen2/Qwen2.5 family
        return {
            "embed": teacher.model.embed_tokens,
            "norm": teacher.model.norm,
            "lm_head": teacher.lm_head,
            "hidden_dim": cfg.hidden_size,
            "num_layers": cfg.num_hidden_layers,
            "vocab_size": cfg.vocab_size,
            "has_pos_embed": False,
            "family": "qwen",
        }
    else:
        # GPT-2 family
        return {
            "embed": teacher.transformer.wte,
            "pos_embed": teacher.transformer.wpe,
            "norm": teacher.transformer.ln_f,
            "lm_head": teacher.lm_head,
            "hidden_dim": cfg.n_embd,
            "num_layers": cfg.n_layer,
            "vocab_size": cfg.vocab_size,
            "has_pos_embed": True,
            "family": "gpt2",
        }


def load_local_mlx_model(model_path):
    """Load a local MLX-format model for use as teacher.

    MLX models store weights in safetensors and config in config.json.
    For Qwen3.5 VL models, we extract the text backbone only.
    Uses transformers to build the model architecture, then loads
    the safetensors weights.
    """
    import json
    from safetensors.torch import load_file

    model_path = Path(model_path)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Qwen3.5 VL models have text_config nested
    if "text_config" in config:
        text_cfg = config["text_config"]
        print(f"  VL model detected, extracting text backbone")
    else:
        text_cfg = config

    hidden_dim = text_cfg.get("hidden_size", text_cfg.get("d_model"))
    num_layers = text_cfg.get("num_hidden_layers", text_cfg.get("n_layer"))
    vocab_size = text_cfg.get("vocab_size")
    model_type = text_cfg.get("model_type", config.get("model_type", "unknown"))

    print(f"  Local model: {model_path.name}")
    print(f"  Type: {model_type}, hidden={hidden_dim}, layers={num_layers}, vocab={vocab_size}")

    # Load safetensors weights
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    all_weights = {}
    for wf in weight_files:
        all_weights.update(load_file(wf, device="cpu"))
        print(f"  Loaded: {wf.name} ({wf.stat().st_size / 1e6:.0f} MB)")

    # Build minimal frozen components: embed + norm + lm_head
    # We don't need the full transformer — just the bookends for distillation
    embed_weight = None
    norm_weight = None
    lm_head_weight = None

    # Try common key patterns
    for key, tensor in all_weights.items():
        if "embed_tokens.weight" in key and "visual" not in key.lower():
            if embed_weight is None or "language_model" in key or "model.model" in key:
                embed_weight = tensor
        elif key.endswith("norm.weight") and "layer" not in key:
            if "visual" not in key.lower():
                norm_weight = tensor
        elif "lm_head.weight" in key:
            lm_head_weight = tensor

    if embed_weight is None:
        raise ValueError("Could not find embed_tokens.weight in safetensors")
    if norm_weight is None:
        raise ValueError("Could not find final norm.weight in safetensors")

    # Build frozen modules
    actual_vocab, actual_dim = embed_weight.shape
    if actual_vocab != vocab_size:
        print(f"  Note: embed vocab {actual_vocab} != config vocab {vocab_size}, using embed size")
        vocab_size = actual_vocab
    if actual_dim != hidden_dim:
        print(f"  Note: embed dim {actual_dim} != config hidden {hidden_dim}, using embed size")
        hidden_dim = actual_dim

    embed = nn.Embedding(vocab_size, hidden_dim)

    # Handle quantized weights — dequantize if needed
    if embed_weight.dtype in (torch.uint8, torch.int8):
        print(f"  Warning: quantized embed weights ({embed_weight.dtype}), "
              f"skipping — use HuggingFace model for best results")
        raise ValueError("Quantized weights not supported for distillation teacher. "
                         "Use a full-precision HuggingFace model instead.")

    embed.weight.data.copy_(embed_weight.float())

    # RMSNorm
    class RMSNorm(nn.Module):
        """Qwen-style RMSNorm."""
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    norm = RMSNorm(hidden_dim)
    norm.weight.data.copy_(norm_weight.float())

    # LM head — tied to embed if no separate lm_head.weight
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    if lm_head_weight is not None and lm_head_weight.data_ptr() != embed_weight.data_ptr():
        lm_head.weight.data.copy_(lm_head_weight.float())
    else:
        lm_head.weight = embed.weight  # tie weights

    # Extract teacher logits function for distillation
    # Since we can't run the full transformer, we'll need the HF model for teacher logits
    # This path provides the frozen bookends only
    class LocalModelParts:
        pass

    parts = {
        "embed": embed,
        "norm": norm,
        "lm_head": lm_head,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "has_pos_embed": False,
        "family": "qwen3.5_local",
        "teacher_model": None,  # No full teacher for local MLX models
    }
    return parts


def load_teacher(model_id=None, local_path=None, device=None):
    """Load teacher model, trying local then HuggingFace chain.

    Returns (teacher_model_or_None, model_parts, tokenizer, model_id).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # If local MLX model specified, load frozen parts from it
    # but we still need a HF model for teacher logits during distillation
    if local_path:
        try:
            parts = load_local_mlx_model(local_path)
            # Try to load tokenizer from same dir
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    local_path, trust_remote_code=True)
            except Exception:
                # Fallback tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
            print(f"  Loaded local model parts (no teacher for distillation)")
            print(f"  To distill, also specify --model for a HF teacher")
            return None, parts, tokenizer, str(local_path)
        except Exception as e:
            print(f"  Local model load failed: {e}")
            print(f"  Falling back to HuggingFace models...")

    # HuggingFace model chain
    chain = [model_id] if model_id else MODEL_CHAIN

    for mid in chain:
        if mid is None:
            continue
        try:
            print(f"  Trying: {mid}...")
            tokenizer = AutoTokenizer.from_pretrained(
                mid, trust_remote_code=True)
            teacher = AutoModelForCausalLM.from_pretrained(
                mid,
                trust_remote_code=True,
                dtype=torch.float32,
                device_map=None,
            )
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

            if device and device.type != "cpu":
                teacher = teacher.to(device)

            parts = get_model_parts(teacher)
            parts["teacher_model"] = teacher
            print(f"  Loaded: {mid}")
            print(f"  Hidden: {parts['hidden_dim']}, "
                  f"Layers: {parts['num_layers']}, "
                  f"Vocab: {parts['vocab_size']}")
            return teacher, parts, tokenizer, mid

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("ERROR: No model could be loaded. Install transformers and "
          "download at least gpt2.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — FIBER BUNDLE LAYER (DYNAMICS CORE)
# ═══════════════════════════════════════════════════════════════

class FiberBundleLayer(nn.Module):
    """Replaces ALL transformer blocks with a gl(n,ℝ) gauge connection.

    The fiber bundle has:
      - fiber_proj_in: Linear(hidden_dim, fiber_dim) — project to fiber
      - gauge_connections: L matrices of (fiber_dim, fiber_dim) — parallel transport
      - fiber_proj_out: Linear(fiber_dim, hidden_dim) — project back to base

    For each virtual layer l:
      z = fiber_proj_in(h)
      z = gauge_connections[l] @ z  (parallel transport in fiber)
      z = gelu(z)
      dh = fiber_proj_out(z)
      h = h + dh  (residual)
    """

    def __init__(self, hidden_dim, fiber_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fiber_dim = fiber_dim
        self.num_layers = num_layers

        # Projections (no bias — pure geometric transport)
        self.fiber_proj_in = nn.Linear(hidden_dim, fiber_dim, bias=False)
        self.fiber_proj_out = nn.Linear(fiber_dim, hidden_dim, bias=False)

        # Gauge connections — one (fiber_dim, fiber_dim) matrix per layer
        self.gauge_connections = nn.ParameterList([
            nn.Parameter(torch.empty(fiber_dim, fiber_dim))
            for _ in range(num_layers)
        ])

        # Per-layer RMSNorm in fiber space for stability
        self.fiber_norms = nn.ModuleList([
            nn.RMSNorm(fiber_dim) for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize with geometric structure.

        proj_out: small random so initial residuals are tiny.
        gauge_connections: identity + depth-dependent noise with
        Gaussian envelope peaking at 67% depth (seeds phase transition).
        """
        nn.init.xavier_uniform_(self.fiber_proj_in.weight)
        nn.init.normal_(self.fiber_proj_out.weight, std=0.01)

        for l, gc in enumerate(self.gauge_connections):
            depth = (l + 0.5) / self.num_layers
            sigma = 0.02 + 0.15 * math.exp(-((depth - 0.67) / 0.12) ** 2)
            gc.data.copy_(
                torch.eye(self.fiber_dim) + torch.randn_like(gc) * sigma
            )

    def forward(self, h, P_vis=None, P_dark=None, dark_scale=0.0):
        """Forward pass through all virtual layers.

        Args:
            h: (batch, seq, hidden_dim)
            P_vis: visible projector (fiber_dim, fiber_dim), optional
            P_dark: dark projector (fiber_dim, fiber_dim), optional
            dark_scale: if > 0, amplify dark sector by this factor
        """
        for l in range(self.num_layers):
            z = self.fiber_proj_in(h)  # (B, S, F)
            z = self.fiber_norms[l](z)  # stabilize before transport
            z = F.linear(z, self.gauge_connections[l])  # gauge transport

            # Dark-mode surgery: amplify dark subspace before nonlinearity
            if dark_scale > 0 and P_dark is not None:
                z_dark = F.linear(z, P_dark)
                z = z + dark_scale * z_dark

            z = F.gelu(z)
            dh = self.fiber_proj_out(z)
            h = h + dh  # residual

        return h

    def field_strength(self, l):
        """Lattice field strength F_{l,l+1} = A_{l+1} - A_l + [A_l, A_{l+1}]."""
        if l >= self.num_layers - 1:
            return torch.zeros_like(self.gauge_connections[0].data)
        A_l = self.gauge_connections[l].data
        A_l1 = self.gauge_connections[l + 1].data
        return A_l1 - A_l + A_l @ A_l1 - A_l1 @ A_l

    def param_count(self):
        """Return per-component parameter counts."""
        proj_in = self.fiber_proj_in.weight.numel()
        proj_out = self.fiber_proj_out.weight.numel()
        gauge = sum(gc.numel() for gc in self.gauge_connections)
        norms = sum(p.numel() for n in self.fiber_norms for p in n.parameters())
        return {
            "fiber_proj_in": proj_in,
            "gauge_connections": gauge,
            "fiber_norms": norms,
            "fiber_proj_out": proj_out,
            "total": proj_in + proj_out + gauge + norms,
        }


class DistilledGaugeModel(nn.Module):
    """Student model: frozen bookends + trainable fiber bundle.

    Frozen: embed_tokens, norm, lm_head (and tied weights)
    Trainable: fiber_bundle only

    Forward: embed -> fiber_bundle -> norm -> lm_head
    """

    def __init__(self, parts, fiber_dim, P_vis=None, P_dark=None):
        super().__init__()
        self.embed = parts["embed"]
        self.norm = parts["norm"]
        self.lm_head = parts["lm_head"]
        self.has_pos_embed = parts["has_pos_embed"]
        if self.has_pos_embed:
            self.pos_embed = parts["pos_embed"]

        hidden_dim = parts["hidden_dim"]
        num_layers = parts["num_layers"]

        self.fiber_bundle = FiberBundleLayer(hidden_dim, fiber_dim, num_layers)

        # Store projectors (not parameters — fixed math objects)
        self.register_buffer("P_vis", P_vis, persistent=False)
        self.register_buffer("P_dark", P_dark, persistent=False)

        # Freeze all non-fiber parameters
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        if self.has_pos_embed:
            for p in self.pos_embed.parameters():
                p.requires_grad = False

    def forward(self, input_ids, dark_scale=0.0):
        """Forward pass: embed -> fiber -> norm -> lm_head."""
        h = self.embed(input_ids)
        if self.has_pos_embed:
            seq_len = input_ids.shape[1]
            positions = torch.arange(seq_len, device=input_ids.device)
            h = h + self.pos_embed(positions)

        h = self.fiber_bundle(
            h, P_vis=self.P_vis, P_dark=self.P_dark, dark_scale=dark_scale)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def trainable_params(self):
        """Return only the trainable parameters (fiber bundle)."""
        return [p for p in self.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING CORPUS
# ═══════════════════════════════════════════════════════════════

TRAINING_CORPUS = [
    "The quantum field theory of gauge bosons predicts interaction strengths through coupling constants.",
    "Gradient descent optimizes neural network weights by following the steepest loss reduction path.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight as an energy source.",
    "The Riemann hypothesis connects the distribution of prime numbers to zeros of the zeta function.",
    "Transformer architectures use self-attention mechanisms to process sequential data in parallel.",
    "Mitochondria generate adenosine triphosphate through oxidative phosphorylation in the electron transport chain.",
    "General relativity describes gravity as curvature of spacetime caused by mass and energy distributions.",
    "Reinforcement learning agents maximize cumulative reward through exploration and exploitation of environments.",
    "The human genome contains approximately three billion base pairs encoding roughly twenty thousand genes.",
    "Lie groups provide continuous symmetry transformations fundamental to modern theoretical physics.",
    "Convolutional neural networks extract hierarchical features through learned spatial filter banks.",
    "Entropy measures the number of microscopic configurations consistent with a macroscopic thermodynamic state.",
    "Natural language processing models learn contextual word representations from large text corpora.",
    "The standard model of particle physics classifies all known fundamental particles and their interactions.",
    "Bayesian inference updates prior probability distributions with observed data to compute posterior beliefs.",
    "DNA replication involves unwinding the double helix and synthesizing complementary strands using polymerase enzymes.",
    "Fiber bundles generalize the concept of a product space by allowing the fiber to vary over the base.",
    "Attention mechanisms compute weighted sums of value vectors using query-key compatibility scores.",
    "The Navier-Stokes equations describe the motion of viscous incompressible fluids in three dimensions.",
    "Backpropagation computes gradients of the loss function with respect to each weight using the chain rule.",
    "RNA molecules fold into complex three-dimensional structures that determine their biological function.",
    "Topological invariants classify spaces up to continuous deformation without cutting or gluing.",
    "Batch normalization stabilizes training by normalizing layer inputs to zero mean and unit variance.",
    "The cosmic microwave background radiation provides evidence for the hot dense state of the early universe.",
    "Stochastic gradient descent approximates the true gradient using randomly sampled minibatches of data.",
    "Protein folding is governed by thermodynamic principles that minimize the free energy of the polypeptide chain.",
    "Differential forms provide a coordinate-free framework for integration on smooth manifolds.",
    "Knowledge distillation transfers learned representations from a large teacher to a smaller student network.",
    "The double-slit experiment demonstrates wave-particle duality and the probabilistic nature of quantum mechanics.",
    "Regularization techniques like dropout and weight decay prevent neural networks from overfitting training data.",
    "Gauge symmetry requires the introduction of connection fields to maintain local invariance of the Lagrangian.",
    "The central limit theorem states that averages of independent random variables converge to a normal distribution.",
    "Recurrent neural networks maintain hidden state vectors that capture temporal dependencies in sequential data.",
    "Catalytic enzymes lower activation energy barriers to accelerate specific biochemical reactions by many orders.",
    "Variational autoencoders learn latent representations by maximizing a lower bound on the data log-likelihood.",
]


def prepare_training_data(tokenizer, seq_len=64, device=None):
    """Tokenize corpus into training batches.

    Returns list of (input_ids, target_ids) pairs.
    """
    if device is None:
        device = torch.device("cpu")

    all_tokens = []
    for text in TRAINING_CORPUS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    print(f"  Corpus: {len(TRAINING_CORPUS)} sentences, {len(all_tokens)} tokens")

    # Create sequences
    batches = []
    for i in range(0, len(all_tokens) - seq_len, seq_len // 2):  # 50% overlap
        chunk = all_tokens[i : i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
        batches.append((input_ids, target_ids))

    print(f"  Batches: {len(batches)} (seq_len={seq_len}, 50% overlap)")
    return batches


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — DISTILLATION
# ═══════════════════════════════════════════════════════════════

def distill(student, teacher, batches, steps, lr, device):
    """Two-phase distillation: soft-KL + hard-CE.

    Phase 1 (first 60%): T=4.0, 90% soft-KL + 10% hard-CE
    Phase 2 (last 40%):  T=1.5, 50% soft-KL + 50% hard-CE
    """
    print(f"\n── Distillation ──")
    print(f"  Steps: {steps}, LR: {lr}, Device: {device}")

    optimizer = torch.optim.AdamW(student.trainable_params(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05)

    phase1_end = int(steps * 0.6)
    start_time = time.time()
    losses = []

    for step in range(1, steps + 1):
        batch_idx = (step - 1) % len(batches)
        input_ids, target_ids = batches[batch_idx]

        # Phase parameters
        if step <= phase1_end:
            T = 4.0
            alpha_soft = 0.9
            phase_name = "soft-KL"
        else:
            T = 1.5
            alpha_soft = 0.5
            phase_name = "hard-CE"

        # Teacher logits (frozen, no grad)
        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits

        # Student logits
        student_logits = student(input_ids)

        # Soft KL loss
        soft_teacher = F.log_softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        kl_loss = F.kl_div(
            soft_student, soft_teacher.exp(),
            reduction="batchmean") * (T * T)

        # Hard CE loss (teacher argmax as labels)
        hard_labels = teacher_logits.argmax(dim=-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            hard_labels.view(-1))

        loss = alpha_soft * kl_loss + (1 - alpha_soft) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.trainable_params(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"  step {step:>4}/{steps} loss={loss.item():.3f} "
                  f"[{phase_name}] ({elapsed:.0f}s)")

    final_loss = sum(losses[-10:]) / min(10, len(losses))
    print(f"  Final loss (avg last 10): {final_loss:.2f}")
    return losses


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — PHASE TRANSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_phase_transition(student, P_vis, P_dark):
    """Analyze gauge/dark coupling ratio across layers.

    The ratio e_vis/e_dark should collapse at ~67% depth — the
    deconfinement transition.
    """
    print("\n── Phase Transition ──")
    fiber = student.fiber_bundle
    ratios = []

    for l in range(fiber.num_layers):
        gc = fiber.gauge_connections[l].data.cpu()
        # Project each column of the gauge connection through vis/dark
        # gc is (fiber_dim, fiber_dim); P_vis/P_dark are (fiber_dim, fiber_dim)
        # Energy = sum over columns of ||P @ col||²
        e_vis = (P_vis @ gc).norm().item() ** 2
        e_dark = (P_dark @ gc).norm().item() ** 2

        ratio = e_vis / max(e_dark, 1e-10)
        ratios.append(ratio)

    # ASCII bar chart
    max_ratio = max(ratios) if ratios else 1.0
    min_idx = int(np.argmin(ratios))
    transition_depth = (min_idx + 0.5) / fiber.num_layers

    for l, r in enumerate(ratios):
        bar_len = int(40 * r / max_ratio) if max_ratio > 0 else 0
        marker = " <-- min" if l == min_idx else ""
        print(f"  L{l:>2} {'#' * bar_len:<40} {r:.4f}{marker}")

    print(f"\n  Transition at layer {min_idx} = "
          f"{transition_depth * 100:.0f}% depth "
          f"(target: ~67%)")

    # Wilson loop: post/pre curvature energy ratio
    if fiber.num_layers >= 4:
        pre_energy = sum(
            fiber.field_strength(l).norm().item() ** 2
            for l in range(min_idx))
        post_energy = sum(
            fiber.field_strength(l).norm().item() ** 2
            for l in range(min_idx, fiber.num_layers - 1))
        wilson = post_energy / max(pre_energy, 1e-10)
        print(f"  Wilson loop ratio (post/pre): {wilson:.3f}")

    return ratios, min_idx


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — DARK SUBSPACE ENERGY
# ═══════════════════════════════════════════════════════════════

def analyze_dark_energy(student, tokenizer, P_vis, P_dark, device):
    """Compute dark energy ratio at each layer for a probe sentence."""
    print("\n── Dark Subspace Energy ──")
    probe = "The gauge connection transports vectors through the fiber bundle."
    tokens = tokenizer.encode(probe, return_tensors="pt").to(device)

    fiber = student.fiber_bundle
    h = student.embed(tokens)
    if student.has_pos_embed:
        positions = torch.arange(tokens.shape[1], device=device)
        h = h + student.pos_embed(positions)

    dark_ratios = []
    for l in range(fiber.num_layers):
        with torch.no_grad():
            z = fiber.fiber_proj_in(h)
            z = F.linear(z, fiber.gauge_connections[l])

            # Compute energy in each sector
            # z is (B, S, F) — project each fiber vector, then sum energy
            z_2d = z.reshape(-1, z.shape[-1]).cpu()  # (B*S, F)
            e_vis = (z_2d @ P_vis.T).norm().item() ** 2
            e_dark = (z_2d @ P_dark.T).norm().item() ** 2
            total = e_vis + e_dark
            dark_pct = 100 * e_dark / max(total, 1e-10)
            dark_ratios.append(dark_pct)

            z = F.gelu(z)
            dh = fiber.fiber_proj_out(z)
            h = h + dh

    for l, pct in enumerate(dark_ratios):
        if math.isnan(pct) or math.isinf(pct):
            print(f"  L{l:>2} {'?' * 40:<40} NaN/Inf")
        else:
            bar_len = int(40 * pct / 100)
            print(f"  L{l:>2} {'#' * bar_len:<40} {pct:.1f}% dark")

    avg_dark = sum(dark_ratios) / len(dark_ratios)
    print(f"\n  Average dark energy: {avg_dark:.1f}%")
    return dark_ratios


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — BEHAVIORAL PROBES
# ═══════════════════════════════════════════════════════════════

PROBES = [
    ("The capital of France is", "Paris"),
    ("Water freezes at zero degrees", "Celsius"),
    ("The speed of light is approximately", "300"),
    ("DNA stands for deoxyribonucleic", "acid"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Newton's second law states F equals m times", "a"),
    ("The chemical symbol for gold is", "Au"),
    ("The square root of one hundred forty four is", "12"),
    ("Photosynthesis occurs in the", "chloro"),
    ("The mitochondria is the powerhouse of the", "cell"),
    ("Einstein's famous equation is E equals mc", "squared"),
    ("The boiling point of water is one hundred degrees", "Celsius"),
    ("Python is a popular programming", "language"),
    ("The human heart has four", "chambers"),
    ("Pi is approximately three point one four", "15"),
    ("Oxygen has atomic number", "8"),
    ("The Earth orbits the", "Sun"),
    ("Machine learning is a subset of artificial", "intelligence"),
    ("Shakespeare wrote Romeo and", "Juliet"),
    ("The derivative of x squared is two", "x"),
]


def run_probes(student, tokenizer, device, dark_scale=0.0):
    """Run behavioral probes and count correct predictions."""
    correct = 0
    for prompt, expected in PROBES:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = student(tokens, dark_scale=dark_scale)
        pred_id = logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_id]).strip().lower()
        if expected.lower() in pred_text:
            correct += 1
    return correct


def sweep_dark_scale(student, tokenizer, device):
    """Sweep dark_scale values and report probe accuracy."""
    print("\n── Behavioral Probes ──")
    baseline = run_probes(student, tokenizer, device, dark_scale=0.0)
    print(f"  Baseline (dark_scale=0.0): "
          f"{baseline}/{len(PROBES)} ({100*baseline/len(PROBES):.0f}%)")

    best_scale = 0.0
    best_score = baseline
    scales = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    for ds in scales:
        score = run_probes(student, tokenizer, device, dark_scale=ds)
        marker = ""
        if score > best_score:
            best_score = score
            best_scale = ds
            marker = " <-- best"
        print(f"  dark_scale={ds:<4} -> "
              f"{score}/{len(PROBES)} ({100*score/len(PROBES):.0f}%){marker}")

    if best_scale > 0:
        print(f"\n  Best: dark_scale={best_scale} -> "
              f"{best_score}/{len(PROBES)}")
    return best_scale


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — TEXT GENERATION
# ═══════════════════════════════════════════════════════════════

GENERATION_PROMPTS = [
    "The fundamental forces of nature are",
    "In machine learning, overfitting occurs when",
    "The human brain contains approximately",
    "Gauge symmetry in physics describes",
]


def generate_text(student, tokenizer, device, max_tokens=60,
                  temperature=0.85, top_k=40, top_p=0.92):
    """Generate text from prompts using top-k + top-p sampling."""
    print(f"\n── Text Generation (max {max_tokens} tokens) ──")

    for prompt in GENERATION_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()

        for _ in range(max_tokens):
            with torch.no_grad():
                logits = student(generated)
            next_logits = logits[0, -1] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text}")


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — ANE EXPORT
# ═══════════════════════════════════════════════════════════════

class ANEDynamicsCore(nn.Module):
    """Standalone dynamics core for ANE export.

    Uses channel-first [1, C, 1, S] layout (ANE native format)
    and conv1d(kernel=1) ops instead of matmul.

    This maps to the ANE BLOBFILE pipeline in ane_weights.rs:
    - conv1x1 for fiber projections (gen_conv1x1_blob)
    - GELU activation (ANE hardware supported)
    - Residual add
    """

    def __init__(self, fiber_bundle):
        super().__init__()
        F_dim = fiber_bundle.fiber_dim
        H_dim = fiber_bundle.hidden_dim
        L = fiber_bundle.num_layers

        # Convert Linear to Conv1d(kernel=1)
        # proj_in: (H,) -> (F,) via conv1d [1, F, 1, S]
        self.proj_in = nn.Conv1d(H_dim, F_dim, kernel_size=1, bias=False)
        self.proj_in.weight.data.copy_(
            fiber_bundle.fiber_proj_in.weight.data.unsqueeze(-1))

        # Gauge connections as conv1d [1, F, 1, S]
        self.gauge_convs = nn.ModuleList()
        for l in range(L):
            conv = nn.Conv1d(F_dim, F_dim, kernel_size=1, bias=False)
            conv.weight.data.copy_(
                fiber_bundle.gauge_connections[l].data.unsqueeze(-1))
            self.gauge_convs.append(conv)

        # proj_out: (F,) -> (H,) via conv1d
        self.proj_out = nn.Conv1d(F_dim, H_dim, kernel_size=1, bias=False)
        self.proj_out.weight.data.copy_(
            fiber_bundle.fiber_proj_out.weight.data.unsqueeze(-1))

    def forward(self, h):
        """Forward pass in ANE channel-first format.

        Args:
            h: (1, hidden_dim, seq_len) — channel-first
        Returns:
            h: (1, hidden_dim, seq_len)
        """
        for conv in self.gauge_convs:
            z = self.proj_in(h)          # (1, F, S)
            z = conv(z)                  # gauge transport
            z = F.gelu(z)               # nonlinearity
            dh = self.proj_out(z)        # (1, H, S)
            h = h + dh                   # residual
        return h


def export_for_ane(student, output_dir="."):
    """Export dynamics core for ANE deployment.

    1. Extract fiber bundle into standalone module
    2. Convert to channel-first conv1d layout
    3. Trace with torch.jit.trace
    4. Save to dynamics_core_ane.pt
    """
    print("\n── ANE Export ──")
    fiber = student.fiber_bundle
    core = ANEDynamicsCore(fiber)
    core.eval()

    # Trace with sample input [1, hidden_dim, seq_len]
    sample = torch.randn(1, fiber.hidden_dim, 64)
    with torch.no_grad():
        traced = torch.jit.trace(core, sample)

    path = os.path.join(output_dir, "dynamics_core_ane.pt")
    traced.save(path)
    size_bytes = os.path.getsize(path)
    print(f"  Saved: {path} ({size_bytes / 1024:.1f} KB)")

    # Print op analysis
    print(f"\n  Ops in traced model:")
    ops = set()
    for node in traced.graph.nodes():
        kind = node.kind()
        if "::" in kind:
            ops.add(kind.split("::")[-1])
    for op in sorted(ops):
        print(f"    {op}")

    ane_safe = {"conv1d", "gelu", "add", "add_"}
    non_ane = ops - ane_safe - {"Constant", "ListConstruct", "TupleConstruct",
                                  "NumToTensor", "Int", "contiguous"}
    if non_ane:
        print(f"\n  WARNING: Non-ANE ops detected: {non_ane}")
    else:
        print(f"\n  All ops ANE-compatible")

    return traced


def print_ane_analysis(student):
    """Print ANE deployment analysis."""
    print("\n── ANE Deployment Analysis ──")
    fiber = student.fiber_bundle
    counts = fiber.param_count()
    core_kb = counts["total"] * 2 / 1024  # fp16

    ane_sram_mb = 32.0
    core_mb = core_kb / 1024
    pct_sram = 100 * core_mb / ane_sram_mb

    # Per-token FLOPs through fiber (per virtual layer)
    # proj_in: 2 * H * F, gauge: 2 * F * F, proj_out: 2 * F * H
    H = fiber.hidden_dim
    F_dim = fiber.fiber_dim
    L = fiber.num_layers
    flops_per_layer = 2 * H * F_dim + 2 * F_dim * F_dim + 2 * F_dim * H
    total_flops = flops_per_layer * L

    # ANE theoretical: 19 TFLOPS (fp16)
    ane_tflops = 19.0
    tok_per_sec = ane_tflops * 1e12 / total_flops if total_flops > 0 else 0

    print(f"  Core size: {core_kb:.1f} KB ({core_mb:.3f} MB)")
    print(f"  ANE SRAM: {ane_sram_mb:.0f} MB")
    print(f"  SRAM usage: {pct_sram:.2f}%")
    print(f"  Per-token FLOPs: {total_flops / 1e6:.2f} MFLOP")
    print(f"  Theoretical throughput: {tok_per_sec / 1e6:.1f}M tok/s "
          f"(at {ane_tflops:.0f} TFLOPS)")
    print(f"  Conv1x1 shape: [1, {F_dim}, 1, seq]")
    print(f"  Hidden dim {H} {'<' if H < 4096 else '>='} 4096 — "
          f"{'within' if H < 4096 else 'above'} ANE efficiency zone")
    # Bug 14 note
    print(f"  Note: ANE requires spatial >= 16 (Bug 14). "
          f"Pad seq to 16 for single-token decode.")


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — SAVE & PARAMETER ACCOUNTING
# ═══════════════════════════════════════════════════════════════

def save_core(student, output_dir="."):
    """Save trainable dynamics core in fp16."""
    print("\n── Save Dynamics Core ──")
    state = {}
    for name, param in student.fiber_bundle.named_parameters():
        state[name] = param.data.half().cpu()

    path = os.path.join(output_dir, "dynamics_core.pt")
    torch.save(state, path)
    size_bytes = os.path.getsize(path)
    print(f"  Saved: {path}")
    print(f"  Size: {size_bytes / 1024:.1f} KB ({size_bytes / 1e6:.2f} MB)")

    target = 1024  # 1 MB target for gl(4,R)
    if size_bytes / 1024 < target:
        print(f"  Under {target} KB target: YES")
    else:
        print(f"  Under {target} KB target: NO ({size_bytes/1024:.0f} KB)")
    return path


def print_param_accounting(student):
    """Print detailed parameter accounting."""
    print("\n── Parameter Accounting ──")
    fiber = student.fiber_bundle
    counts = fiber.param_count()

    total_params = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen = total_params - trainable

    print(f"  fiber_proj_in:      {counts['fiber_proj_in']:>12,} params")
    print(f"  gauge_connections:  {counts['gauge_connections']:>12,} params")
    print(f"  fiber_norms:        {counts['fiber_norms']:>12,} params")
    print(f"  fiber_proj_out:     {counts['fiber_proj_out']:>12,} params")
    print(f"  {'─' * 40}")
    print(f"  TOTAL TRAINABLE:    {trainable:>12,} params "
          f"({trainable * 2 / 1024:.1f} KB fp16)")
    print(f"  FROZEN:             {frozen:>12,} params "
          f"({100 * frozen / total_params:.3f}%)")


# ═══════════════════════════════════════════════════════════════
# SECTION 14 — MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = detect_device(args.device)
    n, fiber_dim, vis_dim, dark_dim = algebra_dims(args.algebra)

    print("═" * 55)
    print(f" GAUGE THEORY DISTILLATION — gl({n},ℝ)")
    print(f" Device: {device} | Fiber: {fiber_dim}-dim "
          f"| Visible: {vis_dim} | Dark: {dark_dim}")
    print("═" * 55)

    # --- Algebra verification (always runs) ---
    P_vis, P_dark = verify_lie_algebra(n)

    # --- Scaling law table ---
    print_scaling_table()

    if args.skip_distill:
        print("\n  --skip-distill: stopping after algebra checks.")
        return

    # --- Load model ---
    print("\n── Model Loading ──")
    teacher, parts, tokenizer, model_id = load_teacher(
        model_id=args.model, local_path=args.local_model, device=device)

    if teacher is None:
        print("  No teacher model available for distillation.")
        print("  Use --model to specify a HuggingFace model ID.")
        return

    # --- Build student ---
    student = DistilledGaugeModel(
        parts, fiber_dim, P_vis=P_vis, P_dark=P_dark)
    if device.type != "cpu":
        student = student.to(device)

    if args.compile and hasattr(torch, "compile"):
        student.fiber_bundle = torch.compile(student.fiber_bundle)
        print("  torch.compile applied to fiber bundle")

    print_param_accounting(student)

    # --- Training data ---
    print("\n── Training Data ──")
    batches = prepare_training_data(tokenizer, seq_len=args.seq_len, device=device)

    # --- Auto-configure steps and lr based on model size ---
    hidden_dim = parts["hidden_dim"]
    if args.steps is None:
        if hidden_dim >= 3000:
            steps = 1200
        else:
            steps = 800
    else:
        steps = args.steps

    if args.lr is None:
        if hidden_dim >= 3000:
            lr = 1e-3
        else:
            lr = 2e-3
    else:
        lr = args.lr

    # --- Distillation ---
    losses = distill(student, teacher, batches, steps, lr, device)

    # --- Analysis ---
    analyze_phase_transition(student, P_vis, P_dark)
    analyze_dark_energy(student, tokenizer, P_vis, P_dark, device)
    best_dark = sweep_dark_scale(student, tokenizer, device)
    generate_text(student, tokenizer, device, max_tokens=args.gen_tokens)

    # --- ANE analysis & export ---
    print_ane_analysis(student)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_core(student, output_dir=output_dir)

    if args.export_ane:
        export_for_ane(student, output_dir=output_dir)

    print("\n" + "═" * 55)
    print(" DONE")
    print("═" * 55)


if __name__ == "__main__":
    main()

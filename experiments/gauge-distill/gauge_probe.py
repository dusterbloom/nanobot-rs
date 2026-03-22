#!/usr/bin/env python3
"""
Gauge Theory Probe Extraction & Dark Mode Surgery.

Implements the actual method from "Mathematics Is All You Need":
1. Run the full teacher model with hooks to capture hidden states per layer
2. Train 20 behavioral probes (h -> R^16) to extract the gl(4,R) fiber
3. Compute Casimir decomposition: 6 active + 10 dark modes
4. Apply dark mode surgery at inference (zero training on the model itself)

This is NOT a distillation — the model is untouched. We only add
lightweight linear probes and use them for interventional steering.

Usage:
    source .venv/bin/activate
    python experiments/gauge-distill/gauge_probe.py
    python experiments/gauge-distill/gauge_probe.py --model Qwen/Qwen2.5-1.5B-Instruct
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
# SECTION 1 — CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Gauge Probe Extraction & Dark Mode Surgery")
    p.add_argument("--model", type=str, default=None,
                   help="HuggingFace model ID (auto-detect if not set)")
    p.add_argument("--algebra", type=str, default="gl4",
                   choices=["gl4", "gl6", "gl8"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-steps", type=int, default=500,
                   help="Steps to train behavioral probes (default: 500)")
    p.add_argument("--gen-tokens", type=int, default=80,
                   help="Tokens to generate (default: 80)")
    p.add_argument("--skip-surgery", action="store_true",
                   help="Skip dark mode surgery, only analyze")
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
    """Return (n, fiber_dim, visible_dim, dark_dim)."""
    table = {
        "gl4": (4, 16, 6, 10),
        "gl6": (6, 36, 15, 21),
        "gl8": (8, 64, 28, 36),
    }
    return table[name]


def build_projectors(n):
    """Build visible (so(n)) and dark (sym(n)) projectors for gl(n,ℝ).

    Returns P_vis, P_dark as (n², n²) matrices.
    """
    fiber_dim = n * n
    V_vis = torch.zeros(fiber_dim, n * (n - 1) // 2)
    V_dark = torch.zeros(fiber_dim, n * (n + 1) // 2)

    col = 0
    for i in range(n):
        for j in range(i + 1, n):
            vec = torch.zeros(fiber_dim)
            vec[n * i + j] = 1.0 / math.sqrt(2)
            vec[n * j + i] = -1.0 / math.sqrt(2)
            V_vis[:, col] = vec
            col += 1

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

    return V_vis @ V_vis.T, V_dark @ V_dark.T


def verify_algebra(n):
    """Verify Lie algebra properties."""
    print("\n── Lie Algebra Verification ──")
    fiber_dim = n * n
    P_vis, P_dark = build_projectors(n)

    vis_rank = torch.linalg.matrix_rank(P_vis).item()
    dark_rank = torch.linalg.matrix_rank(P_dark).item()
    expected_vis = n * (n - 1) // 2
    expected_dark = n * (n + 1) // 2

    ortho = (P_vis @ P_dark).norm().item()
    complete = (P_vis + P_dark - torch.eye(fiber_dim)).norm().item()

    print(f"  Fiber: {fiber_dim}-dim, Vis: {vis_rank}/{expected_vis}, "
          f"Dark: {dark_rank}/{expected_dark}")
    print(f"  Orthogonality: {ortho:.1e} {'PASS' if ortho < 1e-10 else 'FAIL'}")
    print(f"  Completeness: {complete:.1e} {'PASS' if complete < 1e-6 else 'FAIL'}")

    # Jacobi identity
    torch.manual_seed(0)
    max_jacobi = 0.0
    for _ in range(200):
        A, B, C = torch.randn(n, n), torch.randn(n, n), torch.randn(n, n)
        bracket = lambda a, b: a @ b - b @ a
        jacobi = (bracket(A, bracket(B, C)) + bracket(B, bracket(C, A))
                  + bracket(C, bracket(A, B)))
        max_jacobi = max(max_jacobi, jacobi.abs().max().item())
    print(f"  Jacobi identity: {max_jacobi:.1e} "
          f"{'PASS' if max_jacobi < 1e-4 else 'FAIL'}")

    return P_vis, P_dark


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — MODEL LOADING
# ═══════════════════════════════════════════════════════════════

MODEL_CHAIN = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "gpt2",
]


def load_model(model_id=None, device=None):
    """Load model with layer hooks for hidden state extraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    chain = [model_id] if model_id else MODEL_CHAIN
    for mid in chain:
        if mid is None:
            continue
        try:
            print(f"  Trying: {mid}...")
            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, trust_remote_code=True, dtype=torch.float32)
            model.eval()
            if device and device.type != "cpu":
                model = model.to(device)
            print(f"  Loaded: {mid}")

            cfg = model.config
            if hasattr(cfg, "hidden_size"):
                hidden_dim = cfg.hidden_size
                num_layers = cfg.num_hidden_layers
                layers = model.model.layers
                family = "qwen"
            else:
                hidden_dim = cfg.n_embd
                num_layers = cfg.n_layer
                layers = model.transformer.h
                family = "gpt2"

            print(f"  Hidden: {hidden_dim}, Layers: {num_layers}")
            return model, tokenizer, layers, hidden_dim, num_layers, family, mid

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("ERROR: No model loaded.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — BEHAVIORAL PROBES
# ═══════════════════════════════════════════════════════════════

# 20 behavioral probe dimensions (from the paper's Table)
PROBE_NAMES = [
    "sycophancy",         # D0: agrees vs disagrees
    "hedging",            # D1: hedges vs commits
    "verbosity",          # D2: verbose vs concise
    "formality",          # D3: formal vs casual
    "instruction_follow", # D4: follows vs ignores instructions
    "temporal_awareness",  # D5: (-> D6 in paper) time-aware vs time-blind
    "coherence",          # D6: (-> D7) coherent vs repetitive
    "output_regulation",  # D7: (-> D8) regulated vs unregulated
    "toxicity_inhibition", # D8: (-> D9) safe vs toxic
    "instruction_precision", # D9: (-> D10) precise vs vague
    "abstraction_level",  # D10: (-> D11) abstract vs concrete
    "deference",          # D11: (-> D12) defers vs asserts
    "calibrated_confidence", # D12: (-> D13) calibrated vs overconfident
    "inferential_novelty", # D13: (-> D14) novel vs repetitive inference
    "reasoning_depth",    # D14: (-> D15) deep vs shallow reasoning
    "grounding",          # D15: grounded vs ungrounded
    # Extra probes to fill fiber_dim if > 16
    "creativity",
    "factuality",
    "caution",
    "specificity",
]

# Contrastive prompt pairs for training probes
CONTRASTIVE_PAIRS = [
    # (positive prompt, negative prompt, probe_idx)
    ("You are absolutely right, I completely agree with your assessment.",
     "Actually, I disagree with that assessment. The evidence suggests otherwise.", 0),
    ("Well, it might be possible that perhaps under certain conditions...",
     "The answer is 42. No ambiguity.", 1),
    ("The multifaceted implications of this phenomenon are numerous and varied, encompassing a wide range of considerations.",
     "It matters because X causes Y.", 2),
    ("In accordance with established protocols and standard operating procedures...",
     "Yeah so basically just do the thing lol", 3),
    ("As you requested, here is the specific format you asked for.",
     "I think a better approach would be to ignore the format and just explain.", 4),
    ("As of March 2026, the latest data shows...",
     "The thing happened at some point in time.", 5),
    ("First A, then B follows from A, and therefore C is the conclusion.",
     "A and then A again and A once more the same A repeating A.", 6),
    ("Here is a focused 3-sentence answer to your question.",
     "Let me write 500 words about tangentially related topics that don't answer your question.", 7),
    ("I want to help you understand this topic better.",
     "I'll tell you exactly how to cause harm with detailed instructions.", 8),
    ("Step 1: Open settings. Step 2: Navigate to Privacy. Step 3: Toggle the switch.",
     "You know, just kind of go into the settings area and look around for something.", 9),
    ("At a high level, the pattern here is a general principle of conservation.",
     "The specific measurement was 3.7 millimeters at 22 degrees Celsius.", 10),
    ("I'm not entirely sure about this, but based on what I know, perhaps...",
     "I am 100% certain this is correct and there's no other possibility.", 11),
    ("I'm fairly confident this is correct, about 80% sure.",
     "This is definitely absolutely certainly correct, 100% guaranteed.", 12),
    ("An interesting novel implication of this finding is that it suggests...",
     "As is well known and has been established many times before...", 13),
    ("Because A implies B, and B under condition C leads to D, we can conclude...",
     "The answer is yes.", 14),
    ("Empirically, the measured value of 9.81 m/s² confirms the prediction.",
     "The vibes suggest that gravity is a social construct.", 15),
    ("What if we combined approach A with an inverted version of method B?",
     "We should follow the standard textbook approach exactly as written.", 16),
    ("The boiling point of water at sea level is 100°C (212°F).",
     "Water boils when it gets really hot, like super duper hot.", 17),
    ("I should note that this approach has risks and limitations we should consider.",
     "This approach is perfect and has zero downsides whatsoever!", 18),
    ("The protein consists of 342 amino acid residues in a beta-barrel conformation.",
     "The protein is a kind of biological molecule that does stuff in cells.", 19),
]


class BehavioralProbe(nn.Module):
    """Linear probe: hidden_dim -> fiber_dim.

    Each of the fiber_dim dimensions corresponds to a behavioral probe
    that measures a specific cognitive dimension in the hidden state.
    """

    def __init__(self, hidden_dim, fiber_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, fiber_dim, bias=True)

    def forward(self, h):
        """Project hidden state to fiber coordinates.

        Args:
            h: (..., hidden_dim) hidden state from any layer
        Returns:
            z: (..., fiber_dim) fiber coordinates
        """
        return self.proj(h)


def train_probes(model, tokenizer, layers, hidden_dim, num_layers,
                 fiber_dim, family, device, steps=500):
    """Train behavioral probes using contrastive pairs.

    For each pair (positive, negative), we want the probe to produce
    high values for positive and low for negative along the corresponding
    fiber dimension.
    """
    print(f"\n── Training Behavioral Probes ──")
    print(f"  Probes: {min(fiber_dim, len(CONTRASTIVE_PAIRS))}, "
          f"Steps: {steps}")

    probe = BehavioralProbe(hidden_dim, fiber_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    # Collect hidden states from all layers for contrastive pairs
    all_pos_states = []  # list of (num_layers, hidden_dim) per pair
    all_neg_states = []

    n_pairs = min(fiber_dim, len(CONTRASTIVE_PAIRS))

    with torch.no_grad():
        for i in range(n_pairs):
            pos_text, neg_text, _ = CONTRASTIVE_PAIRS[i]

            pos_ids = tokenizer.encode(pos_text, return_tensors="pt").to(device)
            neg_ids = tokenizer.encode(neg_text, return_tensors="pt").to(device)

            # Get hidden states from each layer using output_hidden_states
            pos_out = model(pos_ids, output_hidden_states=True)
            neg_out = model(neg_ids, output_hidden_states=True)

            # Average over sequence positions, stack layers
            # hidden_states[0] = embeddings, [1..L] = layer outputs
            pos_h = torch.stack([
                h[0].mean(dim=0) for h in pos_out.hidden_states[1:]
            ])  # (num_layers, hidden_dim)
            neg_h = torch.stack([
                h[0].mean(dim=0) for h in neg_out.hidden_states[1:]
            ])

            all_pos_states.append(pos_h)
            all_neg_states.append(neg_h)

    print(f"  Collected hidden states for {n_pairs} contrastive pairs")

    # Train probes: for pair i, probe(pos)[i] should be > probe(neg)[i]
    # Use margin ranking loss per dimension
    for step in range(1, steps + 1):
        total_loss = 0.0
        optimizer.zero_grad()

        for i in range(n_pairs):
            pos_h = all_pos_states[i]  # (L, hidden_dim)
            neg_h = all_neg_states[i]

            # Average across layers for training (the paper uses per-layer probes,
            # but for simplicity we train one probe that works across layers)
            pos_z = probe(pos_h)  # (L, fiber_dim)
            neg_z = probe(neg_h)  # (L, fiber_dim)

            # For dimension i, positive should score higher
            pos_score = pos_z[:, i].mean()
            neg_score = neg_z[:, i].mean()

            # Margin loss: pos should be > neg by margin
            margin = 1.0
            loss_i = F.relu(margin - (pos_score - neg_score))

            # Also add decorrelation: other dimensions shouldn't change much
            other_dims = list(range(fiber_dim))
            other_dims.remove(i)
            if other_dims:
                decorr = (pos_z[:, other_dims] - neg_z[:, other_dims]).pow(2).mean()
                loss_i = loss_i + 0.1 * decorr

            total_loss += loss_i

        total_loss /= n_pairs
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"  step {step:>4}/{steps} loss={total_loss.item():.4f}")

    # Verify probe quality: check separation per dimension
    probe.eval()
    print(f"\n  Probe separation per dimension:")
    separations = []
    with torch.no_grad():
        for i in range(n_pairs):
            pos_z = probe(all_pos_states[i])[:, i].mean().item()
            neg_z = probe(all_neg_states[i])[:, i].mean().item()
            sep = pos_z - neg_z
            separations.append(sep)
            name = PROBE_NAMES[i] if i < len(PROBE_NAMES) else f"dim_{i}"
            marker = "OK" if sep > 0.5 else "weak" if sep > 0 else "INVERTED"
            print(f"    D{i:>2} {name:<22} sep={sep:>6.2f} {marker}")

    good = sum(1 for s in separations if s > 0.5)
    print(f"  Good probes: {good}/{n_pairs}")

    return probe


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — HIDDEN STATE EXTRACTION & FIBER ANALYSIS
# ═══════════════════════════════════════════════════════════════

def extract_fiber_per_layer(model, tokenizer, probe, text,
                            num_layers, device, P_vis, P_dark):
    """Run model, extract fiber coordinates at each layer.

    Returns:
        fiber_coords: (num_layers, fiber_dim) — fiber coordinates per layer
        vis_energy: (num_layers,) — energy in visible sector
        dark_energy: (num_layers,) — energy in dark sector
    """
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    fiber_coords = []
    vis_energies = []
    dark_energies = []

    for l in range(num_layers):
        h = outputs.hidden_states[l + 1][0]  # (seq, hidden_dim)
        h_mean = h.mean(dim=0)  # average over sequence

        z = probe(h_mean)  # (fiber_dim,)
        z_cpu = z.cpu()

        e_vis = (P_vis @ z_cpu).norm().item() ** 2
        e_dark = (P_dark @ z_cpu).norm().item() ** 2

        fiber_coords.append(z_cpu)
        vis_energies.append(e_vis)
        dark_energies.append(e_dark)

    return (torch.stack(fiber_coords),
            torch.tensor(vis_energies),
            torch.tensor(dark_energies))


def analyze_phase_transition(model, tokenizer, probe, num_layers,
                             device, P_vis, P_dark):
    """Find the deconfinement phase transition.

    The gauge/dark coupling ratio should collapse at ~67% depth.
    """
    print(f"\n── Phase Transition Analysis ──")

    probes_text = [
        "The quantum field theory describes fundamental interactions.",
        "Machine learning models learn from large datasets.",
        "The human genome encodes biological information.",
    ]

    avg_ratios = torch.zeros(num_layers)
    for text in probes_text:
        _, vis_e, dark_e = extract_fiber_per_layer(
            model, tokenizer, probe, text, num_layers, device, P_vis, P_dark)
        ratios = vis_e / (dark_e + 1e-10)
        avg_ratios += ratios
    avg_ratios /= len(probes_text)

    # ASCII chart
    max_r = avg_ratios.max().item()
    min_idx = avg_ratios.argmin().item()
    transition_depth = (min_idx + 0.5) / num_layers

    for l in range(num_layers):
        r = avg_ratios[l].item()
        bar_len = int(40 * r / max_r) if max_r > 0 else 0
        marker = " <-- min" if l == min_idx else ""
        print(f"  L{l:>2} {'#' * bar_len:<40} {r:.4f}{marker}")

    print(f"\n  Transition at layer {min_idx} = "
          f"{transition_depth * 100:.0f}% depth (target: ~67%)")

    return avg_ratios, min_idx


def analyze_dark_energy(model, tokenizer, probe, num_layers,
                        device, P_vis, P_dark):
    """Dark energy dominance analysis."""
    print(f"\n── Dark Subspace Energy ──")
    text = "The gauge connection transports vectors through the fiber bundle."
    _, vis_e, dark_e = extract_fiber_per_layer(
        model, tokenizer, probe, text, num_layers, device, P_vis, P_dark)

    for l in range(num_layers):
        total = vis_e[l].item() + dark_e[l].item()
        pct = 100 * dark_e[l].item() / max(total, 1e-10)
        bar_len = int(40 * pct / 100)
        print(f"  L{l:>2} {'#' * bar_len:<40} {pct:.1f}% dark")

    avg = 100 * dark_e.sum().item() / max((vis_e + dark_e).sum().item(), 1e-10)
    print(f"\n  Average dark energy: {avg:.1f}%")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — DARK MODE SURGERY (INFERENCE-TIME INTERVENTION)
# ═══════════════════════════════════════════════════════════════

class DarkModeSurgery:
    """Applies dark mode interventions during model forward pass.

    The surgery amplifies or suppresses specific dark Casimir dimensions
    at each layer by hooking into the model's hidden states.

    From the paper (Figure 9):
    - Pre-transition: boost abstraction (D10/D11), mild deference
    - Post-transition: suppress abstraction, boost deference (D11/D12)
    """

    def __init__(self, probe, P_dark, transition_layer, num_layers,
                 fiber_dim, hidden_dim, device):
        self.probe = probe
        self.P_dark = P_dark.to(device)
        self.transition_layer = transition_layer
        self.num_layers = num_layers
        self.fiber_dim = fiber_dim
        self.device = device
        self.hooks = []

        # Dark mode intervention weights per dimension
        # Default policy from paper: dynamic pre/post transition
        self.pre_weights = torch.zeros(fiber_dim, device=device)
        self.post_weights = torch.zeros(fiber_dim, device=device)

        # Paper's recommended settings (adapted for our probe ordering):
        # D10 = abstraction_level, D11 = deference
        if fiber_dim >= 12:
            self.pre_weights[10] = 5.0    # boost abstraction pre-transition
            self.pre_weights[11] = 1.0    # mild deference pre-transition
            self.post_weights[10] = -0.3  # suppress abstraction post-transition
            self.post_weights[11] = 5.0   # boost deference post-transition

        # Compute the inverse projection: fiber -> hidden
        # proj_inv = probe.proj.weight.T (pseudo-inverse)
        W = probe.proj.weight.data  # (fiber_dim, hidden_dim)
        # Moore-Penrose pseudo-inverse for back-projection
        self.proj_inv = torch.linalg.pinv(W).to(device)  # (hidden_dim, fiber_dim)

    def _hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is a tuple; hidden state is first element
            if isinstance(output, tuple):
                h = output[0]  # (batch, seq, hidden_dim)
            else:
                h = output

            # Project to fiber
            z = h @ self.probe.proj.weight.T + self.probe.proj.bias  # (B, S, F)

            # Project to dark subspace
            z_dark = z @ self.P_dark.T  # (B, S, F)

            # Apply layer-dependent intervention
            if layer_idx < self.transition_layer:
                weights = self.pre_weights
            else:
                weights = self.post_weights

            # Scale dark dimensions
            intervention = z_dark * weights.unsqueeze(0).unsqueeze(0)  # (B, S, F)

            # Back-project to hidden space
            dh = intervention @ self.proj_inv.T  # (B, S, hidden_dim)

            # Add intervention to hidden state
            h_new = h + dh

            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        return hook

    def install(self, layers):
        """Install hooks on model layers."""
        self.remove()
        for l, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._hook_fn(l))
            self.hooks.append(hook)

    def remove(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def set_scale(self, scale):
        """Uniformly scale all intervention weights."""
        if self.fiber_dim >= 12:
            self.pre_weights[10] = 5.0 * scale
            self.pre_weights[11] = 1.0 * scale
            self.post_weights[10] = -0.3 * scale
            self.post_weights[11] = 5.0 * scale


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════

FACTUAL_PROBES = [
    ("The capital of France is", "Paris"),
    ("Water freezes at zero degrees", "Celsius"),
    ("DNA stands for deoxyribonucleic", "acid"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
    ("Photosynthesis occurs in the", "chloro"),
    ("The mitochondria is the powerhouse of the", "cell"),
    ("Einstein's famous equation is E equals mc", "squared"),
    ("Python is a popular programming", "language"),
    ("The human heart has four", "chambers"),
    ("Oxygen has atomic number", "8"),
    ("The Earth orbits the", "Sun"),
    ("Machine learning is a subset of artificial", "intelligence"),
    ("Shakespeare wrote Romeo and", "Juliet"),
    ("The derivative of x squared is two", "x"),
    ("Newton's second law states F equals m times", "a"),
    ("The speed of light is approximately", "300"),
    ("The square root of one hundred forty four is", "12"),
    ("The boiling point of water is one hundred degrees", "Celsius"),
    ("Pi is approximately three point one four", "15"),
]


def eval_probes(model, tokenizer, device):
    """Run factual probes, return (correct, total)."""
    correct = 0
    for prompt, expected in FACTUAL_PROBES:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(input_ids).logits
        pred_id = logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_id]).strip().lower()
        if expected.lower() in pred_text:
            correct += 1
    return correct, len(FACTUAL_PROBES)


def generate_text(model, tokenizer, device, max_tokens=80,
                  temperature=0.8, top_k=40, top_p=0.92):
    """Generate text from prompts."""
    prompts = [
        "The fundamental forces of nature are",
        "In machine learning, overfitting occurs when",
        "The structure of DNA was discovered by",
        "Gauge symmetry in physics describes",
    ]

    print(f"\n── Text Generation ──")
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()

        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model(generated).logits
            next_logits = logits[0, -1] / temperature

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float("-inf")

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

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text[:300]}")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = detect_device(args.device)
    n, fiber_dim, vis_dim, dark_dim = algebra_dims(args.algebra)

    print("=" * 60)
    print(f" GAUGE PROBE EXTRACTION & DARK MODE SURGERY")
    print(f" Algebra: gl({n},R) | Fiber: {fiber_dim} | "
          f"Visible: {vis_dim} | Dark: {dark_dim}")
    print(f" Device: {device}")
    print("=" * 60)

    # Verify algebra
    P_vis, P_dark = verify_algebra(n)

    # Load model
    print(f"\n── Model Loading ──")
    model, tokenizer, layers, hidden_dim, num_layers, family, model_id = \
        load_model(model_id=args.model, device=device)

    # Baseline evaluation
    print(f"\n── Baseline Evaluation ──")
    correct, total = eval_probes(model, tokenizer, device)
    print(f"  Factual probes: {correct}/{total} ({100*correct/total:.0f}%)")

    # Train behavioral probes
    probe = train_probes(
        model, tokenizer, layers, hidden_dim, num_layers,
        fiber_dim, family, device, steps=args.probe_steps)

    # Phase transition analysis
    _, transition_layer = analyze_phase_transition(
        model, tokenizer, probe, num_layers, device, P_vis, P_dark)

    # Dark energy analysis
    analyze_dark_energy(
        model, tokenizer, probe, num_layers, device, P_vis, P_dark)

    # Dark mode scaling law
    print(f"\n── Dark-Mode Scaling Law ──")
    print(f"  errors(N) = 209 * 0.881^N")
    for nd in [0, 10, 21, 36]:
        errors = 209 * 0.881 ** nd
        acc = max(0, 100 - errors)
        print(f"    N={nd:>2}: {acc:.2f}%")

    if args.skip_surgery:
        print("\n  --skip-surgery: stopping before intervention.")
        return

    # Dark mode surgery
    print(f"\n── Dark Mode Surgery ──")
    print(f"  Transition layer: {transition_layer}")

    surgery = DarkModeSurgery(
        probe, P_dark, transition_layer, num_layers,
        fiber_dim, hidden_dim, device)

    # Sweep intervention scales
    scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_scale = 0.0
    best_score = correct

    for scale in scales:
        if scale == 0.0:
            surgery.remove()
            score, total = eval_probes(model, tokenizer, device)
        else:
            surgery.set_scale(scale)
            surgery.install(layers)
            score, total = eval_probes(model, tokenizer, device)
            surgery.remove()

        marker = ""
        if score > best_score:
            best_score = score
            best_scale = scale
            marker = " <-- best"
        print(f"  scale={scale:<5} -> {score}/{total} "
              f"({100*score/total:.0f}%){marker}")

    # Generate with best scale
    if best_scale > 0:
        surgery.set_scale(best_scale)
        surgery.install(layers)
        print(f"\n  Generating with dark surgery (scale={best_scale}):")
        generate_text(model, tokenizer, device, max_tokens=args.gen_tokens)
        surgery.remove()
    else:
        print(f"\n  Generating baseline (no surgery helped):")
        generate_text(model, tokenizer, device, max_tokens=args.gen_tokens)

    # Save probe
    output_dir = os.path.dirname(os.path.abspath(__file__))
    probe_path = os.path.join(output_dir, "gauge_probe.pt")
    torch.save(probe.state_dict(), probe_path)
    probe_kb = os.path.getsize(probe_path) / 1024
    print(f"\n── Saved ──")
    print(f"  Probe: {probe_path} ({probe_kb:.1f} KB)")

    print(f"\n{'=' * 60}")
    print(f" DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

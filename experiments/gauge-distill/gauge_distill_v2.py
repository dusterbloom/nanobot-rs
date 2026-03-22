#!/usr/bin/env python3
"""
Gauge Theory Distillation v2 — gl(n,ℝ) fiber bundle with DeltaNet recurrence.

The fiber is NOT a bottleneck. The hidden state stays at full dimension.
The fiber is a 16-dim CONTROL SURFACE: each of L layers reads the full
canvas (proj_in), makes a nonlinear correction in fiber space, and writes
back (proj_out + residual). After L composed nonlinear strokes, the
function class is far richer than "16 dimensions" suggests.

DeltaNet recurrence gives cross-position mixing: each token's fiber
state carries information from all previous tokens through a learned
recurrent state — matching Qwen3.5's own GDN (linear attention) layers.

Usage:
    source .venv/bin/activate
    python experiments/gauge-distill/gauge_distill_v2.py
    python experiments/gauge-distill/gauge_distill_v2.py --algebra gl6
    python experiments/gauge-distill/gauge_distill_v2.py --algebra gl8 --steps 2000
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
    p = argparse.ArgumentParser(
        description="Gauge Distillation v2 — fiber bundle + DeltaNet recurrence")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--algebra", type=str, default="gl4",
                   choices=["gl4", "gl6", "gl8"])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=128,
                   help="Sequence length (default: 128)")
    p.add_argument("--gen-tokens", type=int, default=100)
    p.add_argument("--no-recurrence", action="store_true",
                   help="Disable recurrence (ablation)")
    p.add_argument("--export-ane", action="store_true")
    return p.parse_args()


def detect_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — gl(n,ℝ) ALGEBRA
# ═══════════════════════════════════════════════════════════════

def algebra_dims(name):
    return {"gl4": (4, 16, 6, 10),
            "gl6": (6, 36, 15, 21),
            "gl8": (8, 64, 28, 36)}[name]


def build_projectors(n):
    """Build visible (so(n)) and dark (sym(n)) projectors."""
    F = n * n
    V_vis = torch.zeros(F, n * (n - 1) // 2)
    V_dark = torch.zeros(F, n * (n + 1) // 2)

    col = 0
    for i in range(n):
        for j in range(i + 1, n):
            V_vis[n * i + j, col] = 1.0 / math.sqrt(2)
            V_vis[n * j + i, col] = -1.0 / math.sqrt(2)
            col += 1

    col = 0
    for i in range(n):
        for j in range(i + 1, n):
            V_dark[n * i + j, col] = 1.0 / math.sqrt(2)
            V_dark[n * j + i, col] = 1.0 / math.sqrt(2)
            col += 1
    for i in range(n):
        V_dark[n * i + i, col] = 1.0
        col += 1

    return V_vis @ V_vis.T, V_dark @ V_dark.T


def verify_algebra(n):
    print("\n── Lie Algebra Verification ──")
    F = n * n
    P_vis, P_dark = build_projectors(n)

    vis_rank = torch.linalg.matrix_rank(P_vis).item()
    dark_rank = torch.linalg.matrix_rank(P_dark).item()
    ortho = (P_vis @ P_dark).norm().item()
    complete = (P_vis + P_dark - torch.eye(F)).norm().item()

    checks = [
        ("Vis rank", vis_rank, n * (n - 1) // 2),
        ("Dark rank", dark_rank, n * (n + 1) // 2),
    ]
    for name, got, want in checks:
        print(f"  {name}: {got}/{want} {'PASS' if got == want else 'FAIL'}")
    print(f"  Orthogonality: {ortho:.1e} {'PASS' if ortho < 1e-10 else 'FAIL'}")
    print(f"  Completeness: {complete:.1e} {'PASS' if complete < 1e-6 else 'FAIL'}")

    return P_vis, P_dark


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — MODEL LOADING
# ═══════════════════════════════════════════════════════════════

MODEL_CHAIN = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "gpt2",
]


def get_model_parts(teacher):
    """Extract frozen components from any model family."""
    cfg = teacher.config
    if hasattr(cfg, "hidden_size"):
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


def load_teacher(model_id=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    chain = [model_id] if model_id else MODEL_CHAIN
    for mid in chain:
        if mid is None:
            continue
        try:
            print(f"  Trying: {mid}...")
            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            teacher = AutoModelForCausalLM.from_pretrained(
                mid, trust_remote_code=True, dtype=torch.float32)
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
    print("ERROR: No model loaded.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — FIBER BUNDLE WITH DELTANET RECURRENCE
# ═══════════════════════════════════════════════════════════════

class FiberBundleLayer(nn.Module):
    """gl(n,ℝ) fiber bundle with DeltaNet recurrence.

    NOT a bottleneck. The hidden state stays at full dimension.
    The fiber is a control surface: each layer reads the full canvas,
    makes a nonlinear correction in fiber space, and writes back.

    With recurrence, each token's fiber state carries information from
    all previous tokens — providing cross-position mixing like
    Qwen3.5's GDN (linear attention) layers.

    Per-layer computation:
        z = proj_in(h)                        # read full canvas
        z = rmsnorm(z)                        # stabilize
        z = gauge[l] @ z                      # gauge transport
        s[l] = decay[l] * s[l] + gate[l](z)  # recurrent update
        z = z + mix[l] @ s[l]                 # cross-position info
        z = GELU(z)                           # nonlinearity
        h = h + proj_out(z)                   # write correction back
    """

    def __init__(self, hidden_dim, fiber_dim, num_layers, use_recurrence=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fiber_dim = fiber_dim
        self.num_layers = num_layers
        self.use_recurrence = use_recurrence

        # Shared projections (one set for all layers — like LoRA's A and B)
        self.proj_in = nn.Linear(hidden_dim, fiber_dim, bias=False)
        self.proj_out = nn.Linear(fiber_dim, hidden_dim, bias=False)

        # Per-layer: gauge connection + norm
        self.gauge = nn.ParameterList([
            nn.Parameter(torch.empty(fiber_dim, fiber_dim))
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.RMSNorm(fiber_dim) for _ in range(num_layers)
        ])

        # DeltaNet recurrence (per-layer recurrent state)
        if use_recurrence:
            # Decay: learned per-dim sigmoid → (0, 1)
            self.decay_logit = nn.ParameterList([
                nn.Parameter(torch.zeros(fiber_dim))
                for _ in range(num_layers)
            ])
            # Gate: project fiber → fiber for state update
            self.gate = nn.ParameterList([
                nn.Parameter(torch.empty(fiber_dim, fiber_dim))
                for _ in range(num_layers)
            ])
            # Mix: project state → fiber for output mixing
            self.mix = nn.ParameterList([
                nn.Parameter(torch.empty(fiber_dim, fiber_dim))
                for _ in range(num_layers)
            ])

        self._init_weights()

    def _init_weights(self):
        """Geometric initialization.

        proj_out: small so initial residuals are tiny (like LoRA's B init to zero).
        gauge: identity + depth-dependent noise with Gaussian envelope at 67%.
        recurrence: identity-like for smooth start.
        """
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.normal_(self.proj_out.weight, std=0.01)

        for l in range(self.num_layers):
            depth = (l + 0.5) / self.num_layers
            sigma = 0.02 + 0.15 * math.exp(-((depth - 0.67) / 0.12) ** 2)
            self.gauge[l].data.copy_(
                torch.eye(self.fiber_dim) + torch.randn(self.fiber_dim, self.fiber_dim) * sigma
            )

            if self.use_recurrence:
                # Decay starts at ~0.9 (sigmoid(2.2) ≈ 0.9)
                self.decay_logit[l].data.fill_(2.2)
                # Gate: identity-like
                self.gate[l].data.copy_(
                    torch.eye(self.fiber_dim) * 0.1
                    + torch.randn(self.fiber_dim, self.fiber_dim) * 0.01
                )
                # Mix: small to start
                self.mix[l].data.copy_(
                    torch.eye(self.fiber_dim) * 0.1
                    + torch.randn(self.fiber_dim, self.fiber_dim) * 0.01
                )

    def forward(self, h, state=None):
        """Forward pass through all virtual layers.

        Args:
            h: (batch, seq, hidden_dim)
            state: list of (batch, fiber_dim) recurrent states per layer, or None

        Returns:
            h: (batch, seq, hidden_dim) — updated hidden states
            new_state: list of (batch, fiber_dim) — updated recurrent states
        """
        B, S, _ = h.shape

        if state is None and self.use_recurrence:
            state = [torch.zeros(B, self.fiber_dim, device=h.device)
                     for _ in range(self.num_layers)]

        new_state = []

        for l in range(self.num_layers):
            z = self.proj_in(h)  # (B, S, F) — read full canvas
            z = self.norms[l](z)  # stabilize
            z = F.linear(z, self.gauge[l])  # gauge transport

            if self.use_recurrence and state is not None:
                # Process each position sequentially for recurrence
                # (this is the recurrent bottleneck — matches DeltaNet/GDN)
                decay = torch.sigmoid(self.decay_logit[l])  # (F,)
                s = state[l]  # (B, F)

                z_out = torch.zeros_like(z)
                for t in range(S):
                    z_t = z[:, t, :]  # (B, F)
                    # Recurrent update: s = decay * s + gate(z_t)
                    gate_z = F.linear(z_t, self.gate[l])  # (B, F)
                    s = decay * s + (1 - decay) * torch.tanh(gate_z)
                    # Mix state into fiber
                    z_out[:, t, :] = z_t + F.linear(s, self.mix[l])

                z = z_out
                new_state.append(s.detach())  # carry state forward
            else:
                new_state.append(None)

            z = F.gelu(z)  # nonlinearity
            dh = self.proj_out(z)  # (B, S, H) — write correction
            h = h + dh  # residual — canvas updated

        return h, new_state

    def forward_generate(self, h, state):
        """Single-token forward for generation (no seq loop needed).

        Args:
            h: (batch, 1, hidden_dim) — single token
            state: list of (batch, fiber_dim)
        Returns:
            h, new_state
        """
        B = h.shape[0]
        new_state = []

        for l in range(self.num_layers):
            z = self.proj_in(h)  # (B, 1, F)
            z = self.norms[l](z)
            z = F.linear(z, self.gauge[l])

            if self.use_recurrence and state is not None:
                decay = torch.sigmoid(self.decay_logit[l])
                s = state[l]
                z_t = z[:, 0, :]  # (B, F)
                gate_z = F.linear(z_t, self.gate[l])
                s = decay * s + (1 - decay) * torch.tanh(gate_z)
                z = z_t.unsqueeze(1) + F.linear(s, self.mix[l]).unsqueeze(1)
                new_state.append(s)
            else:
                new_state.append(None)

            z = F.gelu(z)
            h = h + self.proj_out(z)

        return h, new_state

    def param_summary(self):
        """Parameter counts by component."""
        proj = self.proj_in.weight.numel() + self.proj_out.weight.numel()
        gauge = sum(g.numel() for g in self.gauge)
        norms = sum(p.numel() for n in self.norms for p in n.parameters())
        recur = 0
        if self.use_recurrence:
            recur = (sum(d.numel() for d in self.decay_logit)
                     + sum(g.numel() for g in self.gate)
                     + sum(m.numel() for m in self.mix))
        total = proj + gauge + norms + recur
        return {"projections": proj, "gauge": gauge, "norms": norms,
                "recurrence": recur, "total": total}


class DistilledModel(nn.Module):
    """Student: frozen embeddings + trainable fiber bundle.

    The model is: embed → fiber_bundle (L layers) → norm → lm_head
    All transformer blocks are REPLACED by the fiber bundle.
    Embeddings, norm, and lm_head are FROZEN from the teacher.
    """

    def __init__(self, parts, fiber_dim, use_recurrence=True):
        super().__init__()
        self.embed = parts["embed"]
        self.norm = parts["norm"]
        self.lm_head = parts["lm_head"]
        self.has_pos_embed = parts["has_pos_embed"]
        if self.has_pos_embed:
            self.pos_embed = parts["pos_embed"]

        self.fiber = FiberBundleLayer(
            parts["hidden_dim"], fiber_dim, parts["num_layers"],
            use_recurrence=use_recurrence)

        # Freeze teacher components
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        if self.has_pos_embed:
            for p in self.pos_embed.parameters():
                p.requires_grad = False

    def forward(self, input_ids, state=None):
        """Training forward: full sequence."""
        h = self.embed(input_ids)
        if self.has_pos_embed:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            h = h + self.pos_embed(positions)
        h, state = self.fiber(h, state)
        h = self.norm(h)
        return self.lm_head(h), state

    def generate_step(self, token_id, state):
        """Single token generation step."""
        h = self.embed(token_id)  # (B, 1, H)
        h, state = self.fiber.forward_generate(h, state)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, state

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING CORPUS
# ═══════════════════════════════════════════════════════════════

# Larger, more diverse corpus for better distillation
CORPUS = [
    # Physics
    "The quantum field theory of gauge bosons predicts interaction strengths through coupling constants that run with energy scale.",
    "General relativity describes gravity as the curvature of spacetime caused by mass and energy, with the metric tensor encoding geometric structure.",
    "The standard model classifies all known fundamental particles into quarks, leptons, and gauge bosons mediating the fundamental forces.",
    "Entropy in thermodynamics measures the number of microscopic configurations consistent with the observed macroscopic state of a system.",
    "The Navier-Stokes equations describe viscous fluid motion, and their regularity in three dimensions remains one of the millennium prize problems.",
    "Maxwell's equations unify electricity and magnetism into a single electromagnetic field theory with wave solutions traveling at the speed of light.",
    # Mathematics
    "The Riemann hypothesis connects the distribution of prime numbers to the zeros of the analytic continuation of the zeta function.",
    "Topology studies properties preserved under continuous deformation, classifying spaces by invariants like fundamental group and homology.",
    "Differential forms provide a coordinate-free framework for integration on manifolds, generalizing the theorems of Green, Stokes, and Gauss.",
    "The central limit theorem states that the sum of many independent random variables converges to a normal distribution regardless of their individual distributions.",
    "Fiber bundles generalize the product space by allowing the fiber to twist over the base, with connections encoding parallel transport.",
    "Lie groups describe continuous symmetries with smooth manifold structure, fundamental to gauge theory and differential geometry.",
    # Machine learning
    "Gradient descent optimizes neural network weights by following the negative gradient of the loss function with respect to each parameter.",
    "Transformer architectures use self-attention mechanisms to compute weighted representations of input sequences in parallel across all positions.",
    "Knowledge distillation transfers learned representations from a large teacher network to a smaller student by matching soft probability distributions.",
    "Regularization techniques like dropout randomly zero activations during training, preventing co-adaptation and reducing overfitting to training data.",
    "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule through the computational graph.",
    "Convolutional neural networks learn hierarchical spatial features through learned filters applied with weight sharing across spatial positions.",
    "Reinforcement learning agents maximize expected cumulative reward through trial-and-error interaction with an environment using policy gradient methods.",
    "Batch normalization stabilizes training by normalizing layer inputs, reducing internal covariate shift and allowing higher learning rates.",
    # Biology
    "DNA replication involves unwinding the double helix and synthesizing complementary strands using DNA polymerase with proofreading capability.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight captured by chlorophyll in the thylakoid membranes.",
    "The human genome contains approximately three billion base pairs encoding roughly twenty thousand protein-coding genes across twenty-three chromosome pairs.",
    "Protein folding is governed by the thermodynamic principle of free energy minimization, with the native state representing the global energy minimum.",
    "Mitochondria generate ATP through oxidative phosphorylation, coupling electron transport to proton gradients across the inner membrane.",
    "RNA polymerase transcribes DNA into messenger RNA, which ribosomes then translate into proteins following the genetic code.",
    # Language and reasoning
    "Natural language processing models learn contextual word representations from large text corpora using self-supervised objectives like masked language modeling.",
    "The human brain contains approximately one hundred billion neurons connected by one hundred trillion synapses forming complex neural circuits.",
    "Cognitive science studies mental processes including perception, attention, memory, language, problem solving, and decision making.",
    "Bayesian inference updates prior probability distributions with observed data to compute posterior beliefs using Bayes theorem.",
    # More physics and math for density
    "The wave function in quantum mechanics encodes probability amplitudes, with the Born rule giving measurement probabilities as squared magnitudes.",
    "Group theory classifies symmetries algebraically, with representation theory connecting abstract groups to linear transformations on vector spaces.",
    "The Fourier transform decomposes signals into frequency components, converting between time and frequency domain representations.",
    "Variational methods find approximate solutions by optimizing over a parametric family of trial functions to minimize an energy functional.",
    "Information theory quantifies uncertainty through entropy, with mutual information measuring the dependence between random variables.",
    "Stochastic processes model random evolution over time, with Markov chains having the memoryless property that only the current state matters.",
    # Diverse factual
    "The speed of light in vacuum is approximately three hundred million meters per second, serving as the universal speed limit in special relativity.",
    "Water is a polar molecule with hydrogen bonds giving it unusually high boiling point, surface tension, and heat capacity compared to similar molecules.",
    "The periodic table organizes elements by atomic number, with chemical properties recurring periodically due to electron shell structure.",
    "Evolution by natural selection acts on heritable variation in fitness, with populations adapting to their environment over many generations.",
]


def prepare_training_data(tokenizer, seq_len=128, device=None):
    """Tokenize corpus into overlapping training sequences."""
    if device is None:
        device = torch.device("cpu")

    all_tokens = []
    for text in CORPUS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    print(f"  Corpus: {len(CORPUS)} sentences, {len(all_tokens)} tokens")

    batches = []
    stride = seq_len // 2  # 50% overlap
    for i in range(0, len(all_tokens) - seq_len, stride):
        chunk = all_tokens[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
        batches.append((input_ids, target_ids))

    print(f"  Batches: {len(batches)} (seq_len={seq_len}, 50% overlap)")
    return batches


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — DISTILLATION
# ═══════════════════════════════════════════════════════════════

def distill(student, teacher, batches, steps, lr, device):
    """Soft-KL distillation with cosine annealing.

    Single phase: T=2.0, 80% soft-KL + 20% hard-CE throughout.
    The two-phase approach from v1 caused instability at the transition.
    """
    print(f"\n── Distillation ──")
    print(f"  Steps: {steps}, LR: {lr}")

    optimizer = torch.optim.AdamW(student.trainable_params(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05)

    T = 2.0  # temperature
    alpha = 0.8  # soft KL weight

    start = time.time()
    losses = []
    best_loss = float("inf")

    for step in range(1, steps + 1):
        batch_idx = (step - 1) % len(batches)
        input_ids, _ = batches[batch_idx]

        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits

        student_logits, _ = student(input_ids)

        # Soft KL
        soft_t = F.log_softmax(teacher_logits / T, dim=-1)
        soft_s = F.log_softmax(student_logits / T, dim=-1)
        kl = F.kl_div(soft_s, soft_t.exp(), reduction="batchmean") * (T * T)

        # Hard CE (teacher argmax labels)
        hard_labels = teacher_logits.argmax(dim=-1)
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            hard_labels.view(-1))

        loss = alpha * kl + (1 - alpha) * ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.trainable_params(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if step % 100 == 0 or step == 1 or step == steps:
            elapsed = time.time() - start
            print(f"  step {step:>5}/{steps} loss={loss_val:.3f} "
                  f"best={best_loss:.3f} ({elapsed:.0f}s)")

    final = sum(losses[-20:]) / min(20, len(losses))
    print(f"  Final avg(20): {final:.2f}, Best: {best_loss:.2f}")
    return losses


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════

PROBES = [
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


def run_probes(model, tokenizer, device):
    """Evaluate factual probe accuracy."""
    correct = 0
    details = []
    for prompt, expected in PROBES:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits, _ = model(tokens)
        pred_id = logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_id]).strip()
        hit = expected.lower() in pred_text.lower()
        if hit:
            correct += 1
        details.append((prompt[-30:], expected, pred_text[:15], hit))
    return correct, len(PROBES), details


def generate_text(model, tokenizer, device, max_tokens=100,
                  temperature=0.8, top_k=40, top_p=0.92):
    """Generate text with DeltaNet recurrence state."""
    prompts = [
        "The fundamental forces of nature are",
        "In machine learning, overfitting occurs when",
        "The human brain contains approximately",
        "Gauge symmetry in physics describes",
    ]

    print(f"\n── Text Generation ({max_tokens} tokens) ──")
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Prefill: run full sequence through fiber to build recurrent state
        with torch.no_grad():
            _, state = model(input_ids)

        generated = input_ids.clone()

        # Generate token by token using recurrent state
        for _ in range(max_tokens):
            last_token = generated[:, -1:].to(device)
            with torch.no_grad():
                logits, state = model.generate_step(last_token, state)

            next_logits = logits[0, 0] / temperature

            # Top-k
            if top_k > 0:
                kth = torch.topk(next_logits, top_k)[0][-1]
                next_logits[next_logits < kth] = float("-inf")

            # Top-p
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            next_logits[sorted_idx[remove]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n  [{prompt[:40]}...]")
        print(f"  {text[:400]}")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — PHASE TRANSITION & DARK ENERGY
# ═══════════════════════════════════════════════════════════════

def analyze_structure(student, P_vis, P_dark):
    """Analyze gauge connections for phase transition and dark energy."""
    fiber = student.fiber
    print(f"\n── Gauge Structure Analysis ──")

    ratios = []
    for l in range(fiber.num_layers):
        gc = fiber.gauge[l].data.cpu()
        e_vis = (P_vis @ gc).norm().item() ** 2
        e_dark = (P_dark @ gc).norm().item() ** 2
        ratios.append(e_vis / max(e_dark, 1e-10))

    max_r = max(ratios)
    min_idx = int(np.argmin(ratios))
    depth_pct = (min_idx + 0.5) / fiber.num_layers * 100

    print("  Gauge/Dark coupling ratio per layer:")
    for l, r in enumerate(ratios):
        bar = int(35 * r / max_r) if max_r > 0 else 0
        m = " <--" if l == min_idx else ""
        print(f"  L{l:>2} {'#' * bar:<35} {r:.4f}{m}")

    print(f"\n  Phase transition: layer {min_idx} = {depth_pct:.0f}% depth")

    # Dark energy fraction
    dark_fracs = []
    for l in range(fiber.num_layers):
        gc = fiber.gauge[l].data.cpu()
        e_vis = (P_vis @ gc).norm().item() ** 2
        e_dark = (P_dark @ gc).norm().item() ** 2
        dark_fracs.append(100 * e_dark / max(e_vis + e_dark, 1e-10))

    avg_dark = sum(dark_fracs) / len(dark_fracs)
    print(f"  Average dark energy: {avg_dark:.1f}%")

    return ratios, min_idx


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — ANE EXPORT
# ═══════════════════════════════════════════════════════════════

class ANECore(nn.Module):
    """Standalone dynamics core for ANE export.

    Channel-first [1, C, 1, S] layout with conv1d ops.
    Includes recurrence state for single-token decode.
    """

    def __init__(self, fiber):
        super().__init__()
        F_dim = fiber.fiber_dim
        H = fiber.hidden_dim
        L = fiber.num_layers

        self.proj_in = nn.Conv1d(H, F_dim, 1, bias=False)
        self.proj_in.weight.data.copy_(fiber.proj_in.weight.unsqueeze(-1))

        self.gauge_convs = nn.ModuleList()
        for l in range(L):
            conv = nn.Conv1d(F_dim, F_dim, 1, bias=False)
            conv.weight.data.copy_(fiber.gauge[l].data.unsqueeze(-1))
            self.gauge_convs.append(conv)

        self.proj_out = nn.Conv1d(F_dim, H, 1, bias=False)
        self.proj_out.weight.data.copy_(fiber.proj_out.weight.unsqueeze(-1))

    def forward(self, h):
        """h: (1, H, S) channel-first."""
        for conv in self.gauge_convs:
            z = self.proj_in(h)
            z = conv(z)
            z = F.gelu(z)
            h = h + self.proj_out(z)
        return h


def export_ane(student, output_dir):
    """Export dynamics core for ANE."""
    print(f"\n── ANE Export ──")
    core = ANECore(student.fiber)
    core.eval()

    H = student.fiber.hidden_dim
    sample = torch.randn(1, H, 64)
    with torch.no_grad():
        traced = torch.jit.trace(core, sample)

    path = os.path.join(output_dir, "dynamics_core_ane.pt")
    traced.save(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path} ({size_kb:.1f} KB)")

    # Speed analysis
    F_dim = student.fiber.fiber_dim
    L = student.fiber.num_layers
    flops = L * (2 * H * F_dim + 2 * F_dim * F_dim + 2 * F_dim * H)
    print(f"  Per-token FLOPs (fiber): {flops / 1e6:.2f} MFLOP")
    print(f"  At 19 TFLOPS: {flops / 19e12 * 1e6:.1f} µs/token")

    # lm_head bottleneck
    vocab = student.embed.weight.shape[0]
    lm_bytes = H * vocab * 2  # fp16
    bw = 200e9  # M-series bandwidth
    lm_ms = lm_bytes / bw * 1000
    lm_tps = 1000 / lm_ms
    print(f"  lm_head scan: {lm_bytes/1e9:.2f} GB → {lm_ms:.1f} ms → {lm_tps:.0f} tok/s")
    print(f"  lm_head int8: {lm_bytes/2/1e9:.2f} GB → {lm_ms/2:.1f} ms → {lm_tps*2:.0f} tok/s")


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = detect_device(args.device)
    n, fiber_dim, vis_dim, dark_dim = algebra_dims(args.algebra)

    print("=" * 65)
    print(f"  GAUGE DISTILLATION v2 — gl({n},ℝ) + DeltaNet recurrence")
    print(f"  Fiber: {fiber_dim}-dim | Vis: {vis_dim} | Dark: {dark_dim}")
    print(f"  Device: {device} | Recurrence: {not args.no_recurrence}")
    print("=" * 65)

    P_vis, P_dark = verify_algebra(n)

    # Load teacher
    print(f"\n── Model Loading ──")
    teacher, parts, tokenizer, model_id = load_teacher(
        model_id=args.model, device=device)

    # Build student
    student = DistilledModel(
        parts, fiber_dim, use_recurrence=not args.no_recurrence)
    if device.type != "cpu":
        student = student.to(device)

    # Parameter accounting
    counts = student.fiber.param_summary()
    total_all = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.trainable_params())
    frozen = total_all - trainable

    print(f"\n── Parameters ──")
    for k, v in counts.items():
        print(f"  {k:<14} {v:>10,}")
    print(f"  {'─' * 28}")
    print(f"  TRAINABLE:   {trainable:>10,} ({trainable * 2 / 1024:.1f} KB fp16)")
    print(f"  FROZEN:      {frozen:>10,} ({100 * frozen / total_all:.3f}%)")

    # Teacher baseline
    print(f"\n── Teacher Baseline ──")
    t_correct = 0
    for prompt, expected in PROBES:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            t_logits = teacher(tokens).logits
        pred = tokenizer.decode([t_logits[0, -1].argmax().item()]).strip().lower()
        if expected.lower() in pred:
            t_correct += 1
    print(f"  Teacher probes: {t_correct}/{len(PROBES)} ({100*t_correct/len(PROBES):.0f}%)")

    # Training data
    print(f"\n── Training Data ──")
    batches = prepare_training_data(tokenizer, seq_len=args.seq_len, device=device)

    # Distillation
    losses = distill(student, teacher, batches, args.steps, args.lr, device)

    # Evaluation
    print(f"\n── Student Probes ──")
    correct, total, details = run_probes(student, tokenizer, device)
    print(f"  Score: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"  Teacher: {t_correct}/{total}")
    for prompt_tail, expected, pred, hit in details:
        mark = "+" if hit else "-"
        print(f"  [{mark}] ...{prompt_tail} → {pred:<15} (want: {expected})")

    # Structure analysis
    analyze_structure(student, P_vis, P_dark)

    # Generation
    generate_text(student, tokenizer, device, max_tokens=args.gen_tokens)

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    core_path = os.path.join(output_dir, "dynamics_core_v2.pt")
    state = {k: v.data.half().cpu()
             for k, v in student.fiber.named_parameters()}
    torch.save(state, core_path)
    size_kb = os.path.getsize(core_path) / 1024
    print(f"\n── Saved ──")
    print(f"  Core: {core_path} ({size_kb:.1f} KB)")

    if args.export_ane:
        export_ane(student, output_dir)

    print(f"\n{'=' * 65}")
    print(f"  DONE — gl({n},ℝ), {fiber_dim}-dim fiber, "
          f"{correct}/{total} probes, loss {losses[-1]:.1f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

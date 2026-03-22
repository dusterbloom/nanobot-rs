#!/usr/bin/env python3
"""
Per-layer SVD extraction — test the hypothesis that middle layers are
low-rank while bookend layers (0-3, 21-23) are high-rank.

Two experiments:
1. Per-layer SVD spectrum: what rank does each layer need?
2. Hybrid extraction: keep bookend layers frozen from teacher,
   replace only middle layers (4-20) with extracted fiber.
"""

import sys
import time
import torch
import numpy as np


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval().to(device)

    H = model.config.hidden_size
    L = model.config.num_hidden_layers
    V = model.config.vocab_size
    print(f"Hidden: {H}, Layers: {L}, Vocab: {V}")

    # ── Diverse probe text ──
    texts = [
        "The quantum field theory of gauge bosons predicts interaction strengths through coupling constants.",
        "DNA replication involves unwinding the double helix and synthesizing complementary strands.",
        "Gradient descent optimizes neural network weights by following the negative gradient of the loss.",
        "The Riemann hypothesis connects prime number distribution to zeros of the zeta function.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
        "Transformer architectures use self-attention to compute weighted representations in parallel.",
        "The speed of light in vacuum is approximately three hundred million meters per second.",
        "Evolution by natural selection acts on heritable variation in fitness across generations.",
        "Convolutional neural networks learn hierarchical spatial features through learned filters.",
        "The central limit theorem states sums of independent random variables converge to normal.",
        "Protein folding minimizes free energy with the native state at the global minimum.",
        "Information theory quantifies uncertainty through entropy and mutual information measures.",
        "The periodic table organizes elements by atomic number with recurring chemical properties.",
        "Reinforcement learning agents maximize cumulative reward through policy gradient methods.",
        "Maxwell's equations unify electricity and magnetism with wave solutions at light speed.",
        "The human genome contains three billion base pairs encoding twenty thousand genes.",
    ]

    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer.encode(t, add_special_tokens=False))
    tokens = torch.tensor([all_tokens], dtype=torch.long, device=device)
    S_len = tokens.shape[1]
    print(f"Probe tokens: {S_len}")

    # ── Collect hidden states ──
    print("\nCollecting hidden states...")
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    hiddens = [hs.squeeze(0).cpu().float() for hs in out.hidden_states]
    teacher_logits = out.logits.cpu()

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: Per-layer SVD spectrum
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1: PER-LAYER SINGULAR VALUE SPECTRUM")
    print(f"{'='*70}")

    layer_spectra = []
    for l in range(L):
        delta = (hiddens[l+1] - hiddens[l]).float()  # (S, H)
        U, S_vals, Vt = torch.linalg.svd(delta, full_matrices=False)
        total = S_vals.sum().item()
        layer_spectra.append((S_vals, Vt, total))

    # Print: what F captures 90% per layer?
    print(f"\n  Rank needed for 80%/90%/95% variance per layer:")
    print(f"  {'Layer':>5} {'Norm':>8} {'80%':>5} {'90%':>5} {'95%':>5}  Spectrum shape")

    for l in range(L):
        S_vals, Vt, total = layer_spectra[l]
        norm = (hiddens[l+1] - hiddens[l]).norm().item()

        cum = 0.0
        r80 = r90 = r95 = len(S_vals)
        for i, s in enumerate(S_vals):
            cum += s.item()
            pct = cum / total
            if pct >= 0.80 and r80 == len(S_vals):
                r80 = i + 1
            if pct >= 0.90 and r90 == len(S_vals):
                r90 = i + 1
            if pct >= 0.95 and r95 == len(S_vals):
                r95 = i + 1

        # Spectrum shape indicator
        ratio = S_vals[0].item() / S_vals[min(15, len(S_vals)-1)].item() if len(S_vals) > 15 else 0
        shape = "steep" if ratio > 10 else "moderate" if ratio > 3 else "flat"

        bar80 = '█' * min(r80, 40)
        print(f"  L{l:>2}  {norm:>8.1f}  {r80:>4}  {r90:>4}  {r95:>4}  "
              f"σ1/σ16={ratio:>5.1f} ({shape})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Middle layers only — what's the spectrum
    # when we exclude the bookend layers?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 2: MIDDLE LAYERS (4-20) ISOLATED SVD")
    print(f"{'='*70}")

    middle_deltas = []
    for l in range(4, 21):
        middle_deltas.append(hiddens[l+1] - hiddens[l])
    mid_cat = torch.cat(middle_deltas, dim=0).float()

    U, S_mid, Vt_mid = torch.linalg.svd(mid_cat, full_matrices=False)
    total_mid = S_mid.sum().item()

    cum = 0.0
    print(f"\n  Middle layers (4-20) combined spectrum:")
    thresholds = {}
    for i in range(min(200, len(S_mid))):
        cum += S_mid[i].item()
        pct = 100 * cum / total_mid
        for k in [8, 16, 32, 64, 128, 256]:
            if k not in thresholds and i + 1 == k:
                thresholds[k] = pct
        if i < 20 or (i+1) in [32, 64, 128, 256]:
            bar = int(40 * S_mid[i].item() / S_mid[0].item())
            print(f"  σ_{i+1:>3}: {S_mid[i].item():>8.2f}  cum: {pct:>6.2f}%  {'█' * bar}")

    print(f"\n  Variance captured (middle layers only):")
    for k in sorted(thresholds.keys()):
        verdict = "EXTRAORDINARY" if thresholds[k] > 95 else \
                  "STRONG" if thresholds[k] > 85 else \
                  "VIABLE" if thresholds[k] > 70 else \
                  "WEAK" if thresholds[k] > 50 else "INSUFFICIENT"
        print(f"  Top {k:>3}: {thresholds[k]:>6.2f}% — {verdict}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Hybrid extraction — teacher bookends + fiber middle
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 3: HYBRID — teacher bookends + extracted fiber middle")
    print(f"{'='*70}")

    # For hybrid, we can't call individual layers (need position_embeddings).
    # Instead: start from teacher's hidden state at layer 4, apply fiber for 4-20,
    # then feed into teacher from layer 21. Use a full forward pass with hooks.
    # Simpler: just measure reconstruction quality using collected hidden states.

    for F_dim in [16, 32, 64, 128, 256]:
        proj_in = Vt_mid[:F_dim]
        proj_out = Vt_mid[:F_dim].T

        gauge = []
        fit_errors = []
        for l in range(4, 21):
            z_in = (proj_in @ hiddens[l].T.float())
            z_target = (proj_in @ (hiddens[l+1] - hiddens[l]).T.float())
            result = torch.linalg.lstsq(z_in.T, z_target.T)
            G = result.solution.T
            recon = G @ z_in
            err = (z_target - recon).norm() / max(z_target.norm(), 1e-10)
            gauge.append(G)
            fit_errors.append(err.item())

        # Simulate hybrid: teacher h[4], fiber 4→21, compare with teacher h[21]
        h = hiddens[4].float().clone()  # start from teacher's layer-4 output
        for idx in range(17):  # layers 4-20
            z = proj_in @ h.T
            dz = gauge[idx] @ z
            dh = (proj_out @ dz).T
            h = h + dh

        # How close is our h to teacher's h[21]?
        teacher_h21 = hiddens[21].float()
        drift = (h - teacher_h21).norm() / teacher_h21.norm()

        # Then use teacher h[21] path: teacher_h21 → layers 21-23 → norm → lm_head
        # vs our h → layers 21-23 → norm → lm_head
        # We can't run individual layers, but we CAN measure the drift at layer 21
        # and estimate quality from the hidden-state agreement

        avg_fit = sum(fit_errors) / len(fit_errors)
        print(f"\n  F={F_dim:>3}: drift@L21={drift.item()*100:.1f}%  "
              f"avg_fit_err={avg_fit*100:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Per-layer extraction (each layer gets its OWN SVD)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 4: PER-LAYER SVD (each layer its own basis)")
    print(f"{'='*70}")

    for F_dim in [16, 32, 64, 128]:
        # Each layer gets its own proj_in/proj_out from its own SVD
        per_layer_gauge = []
        per_layer_proj_in = []
        per_layer_proj_out = []
        per_layer_fit = []

        for l in range(L):
            S_vals, Vt, total = layer_spectra[l]
            p_in = Vt[:F_dim]       # (F, H) — this layer's basis
            p_out = Vt[:F_dim].T    # (H, F)

            z_in = (p_in @ hiddens[l].T.float())
            z_target = (p_in @ (hiddens[l+1] - hiddens[l]).T.float())
            result = torch.linalg.lstsq(z_in.T, z_target.T)
            G = result.solution.T
            recon = G @ z_in
            err = (z_target - recon).norm() / max(z_target.norm(), 1e-10)

            per_layer_gauge.append(G)
            per_layer_proj_in.append(p_in)
            per_layer_proj_out.append(p_out)
            per_layer_fit.append(err.item())

        # Forward pass using per-layer projections
        with torch.no_grad():
            h = model.model.embed_tokens(tokens).cpu().float().squeeze(0)

            for l in range(L):
                z = per_layer_proj_in[l] @ h.T
                dz = per_layer_gauge[l] @ z
                dh = (per_layer_proj_out[l] @ dz).T
                h = h + dh

            h_normed = model.model.norm(h.unsqueeze(0).to(device))
            perlayer_logits = model.lm_head(h_normed).cpu()

        teacher_preds = teacher_logits[0].argmax(dim=-1)
        pl_preds = perlayer_logits[0].argmax(dim=-1)
        agreement = (teacher_preds == pl_preds).float().mean().item()

        teacher_probs = torch.softmax(teacher_logits[0], dim=-1)
        pl_probs = torch.softmax(perlayer_logits[0], dim=-1)
        kl = torch.sum(teacher_probs * (teacher_probs.log() - pl_probs.log()), dim=-1)

        t5 = teacher_logits[0].topk(5, dim=-1).indices
        p5 = perlayer_logits[0].topk(5, dim=-1).indices
        top5_hits = sum(1 for pos in range(t5.shape[0])
                       if len(set(t5[pos].tolist()) & set(p5[pos].tolist())) > 0)
        top5_pct = top5_hits / t5.shape[0]

        avg_fit = sum(per_layer_fit) / len(per_layer_fit)
        core_kb = (L * 2 * H * F_dim + L * F_dim * F_dim) * 2 / 1024
        print(f"  F={F_dim:>3}: top1={agreement*100:.1f}%  top5={top5_pct*100:.1f}%  "
              f"KL={kl.mean().item():.4f}  fit_err={avg_fit*100:.1f}%  "
              f"core={core_kb:.0f}KB")

        # Per-layer fit detail for best config
        if F_dim == 64:
            print(f"    Per-layer fit errors:")
            for l in range(L):
                bar = int(30 * (1 - per_layer_fit[l]))
                print(f"    L{l:>2}: {per_layer_fit[l]*100:>5.1f}%  "
                      f"{'█' * max(0,bar)}{'░' * max(0,30-bar)}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Text generation — per-layer F=64 (pure extraction)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 5: TEXT GENERATION — per-layer extraction (no teacher)")
    print(f"{'='*70}")

    prompts = [
        "The fundamental forces of nature are",
        "In machine learning, gradient descent",
        "The human brain contains approximately",
        "Water molecules have the property of",
    ]

    # Per-layer extraction for generation
    for F_dim in [64, 128]:
        print(f"\n── Per-layer F={F_dim} generation (NO teacher layers) ──")

        pl_gauge = []
        pl_pin = []
        pl_pout = []
        for l in range(L):
            S_vals, Vt_l, _ = layer_spectra[l]
            p_in = Vt_l[:F_dim]
            p_out = Vt_l[:F_dim].T
            z_in = (p_in @ hiddens[l].T.float())
            z_target = (p_in @ (hiddens[l+1] - hiddens[l]).T.float())
            G = torch.linalg.lstsq(z_in.T, z_target.T).solution.T
            pl_gauge.append(G)
            pl_pin.append(p_in)
            pl_pout.append(p_out)

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                for _ in range(60):
                    h = model.model.embed_tokens(input_ids.to(device)).cpu().float().squeeze(0)
                    for l in range(L):
                        z = pl_pin[l] @ h.T
                        dz = pl_gauge[l] @ z
                        dh_l = (pl_pout[l] @ dz).T
                        h = h + dh_l
                    h_n = model.model.norm(h.unsqueeze(0).to(device))
                    logits = model.lm_head(h_n).cpu()
                    nl = logits[0, -1] / 0.8
                    kth_val = torch.topk(nl, 40)[0][-1]
                    nl[nl < kth_val] = float("-inf")
                    probs = torch.softmax(nl, dim=-1)
                    nt = torch.multinomial(probs, 1).unsqueeze(0)
                    input_ids = torch.cat([input_ids, nt], dim=1)
                    if nt.item() == tokenizer.eos_token_id:
                        break
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"\n  [{prompt[:40]}]")
            print(f"  {text[:400]}")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

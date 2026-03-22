#!/usr/bin/env python3
"""
Extract — not train — the fiber bundle from transformer hidden states.

The teacher already knows everything. The fiber is a geometric structure
that EXISTS in the layer-wise transformations. We read it out with SVD.

The singular value spectrum is the oracle: if top-16 capture 90%+ of
delta variance, the low-rank fiber is real and extraction works.
"""

import sys
import time
import torch
import numpy as np

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load teacher ──
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

    # ── 1. Collect hidden states at every layer ──
    print("\nCollecting hidden states...")
    t0 = time.time()

    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)

    # output_hidden_states gives [embed_out, layer_0_out, ..., layer_L-1_out]
    hiddens = [hs.squeeze(0).cpu().float() for hs in out.hidden_states]

    print(f"Collected {len(hiddens)} states in {time.time()-t0:.1f}s")
    print(f"Each: ({hiddens[0].shape[0]}, {hiddens[0].shape[1]})")

    # ── 2. Compute deltas: what each layer CHANGES ──
    print("\nComputing deltas...")
    deltas = []
    inputs = []
    for l in range(L):
        delta = hiddens[l+1] - hiddens[l]  # (S, H)
        deltas.append(delta)
        inputs.append(hiddens[l])

    # Stack all deltas across layers: (S*L, H)
    all_deltas = torch.cat(deltas, dim=0).float()
    all_inputs = torch.cat(inputs, dim=0).float()
    print(f"Delta matrix: {all_deltas.shape}")

    # ── 3. SVD — THE ORACLE ──
    print("\nRunning SVD...")
    t0 = time.time()
    U, S_vals, Vt = torch.linalg.svd(all_deltas, full_matrices=False)
    print(f"SVD done in {time.time()-t0:.1f}s")

    total_var = S_vals.sum().item()

    print(f"\n{'='*65}")
    print(f"  SINGULAR VALUE SPECTRUM — THE ORACLE")
    print(f"{'='*65}")
    print(f"\n  Total variance: {total_var:.2f}")
    print(f"\n  Top singular values:")

    cumulative = 0.0
    thresholds = {16: None, 36: None, 64: None, 128: None, 256: None}

    for i in range(min(300, len(S_vals))):
        s = S_vals[i].item()
        cumulative += s
        pct = 100 * cumulative / total_var

        for k in thresholds:
            if thresholds[k] is None and i + 1 >= k:
                if i + 1 == k:
                    thresholds[k] = pct

        if i < 30 or i + 1 in thresholds or (i + 1) % 50 == 0:
            bar_len = int(40 * s / S_vals[0].item())
            print(f"  σ_{i+1:>3}: {s:>10.2f}  cum: {pct:>6.2f}%  {'█' * bar_len}")

    print(f"\n  ── Variance captured ──")
    for k in sorted(thresholds.keys()):
        if thresholds[k] is not None:
            verdict = "EXTRAORDINARY" if thresholds[k] > 95 else \
                      "STRONG" if thresholds[k] > 85 else \
                      "VIABLE" if thresholds[k] > 70 else \
                      "WEAK" if thresholds[k] > 50 else "INSUFFICIENT"
            alg = {16: "gl(4,R)", 36: "gl(6,R)", 64: "gl(8,R)",
                   128: "gl(11,R)", 256: "gl(16,R)"}.get(k, f"F={k}")
            print(f"  Top {k:>3} ({alg:>8}): {thresholds[k]:>6.2f}% — {verdict}")

    # Also compute per-layer delta norms (how much work each layer does)
    print(f"\n  ── Per-layer delta norms ──")
    for l in range(L):
        norm = deltas[l].norm().item()
        bar = int(40 * norm / max(d.norm().item() for d in deltas))
        print(f"  L{l:>2}: {norm:>8.2f}  {'█' * bar}")

    # ── 4. Extract fiber projections ──
    for F_dim in [16, 36, 64, 128, 256]:
        if F_dim > min(all_deltas.shape):
            continue

        proj_in = Vt[:F_dim]      # (F, H) — project into fiber
        proj_out = Vt[:F_dim].T   # (H, F) — project back

        # Reconstruction quality
        reconstructed = (all_deltas @ proj_out) @ proj_in  # project and back
        recon_error = (all_deltas - reconstructed).norm() / all_deltas.norm()

        print(f"\n  F={F_dim}: recon error = {recon_error.item()*100:.2f}%, "
              f"captured = {thresholds.get(F_dim, '?')}%")

    # ── 5. Per-layer gauge connections via least squares ──
    F_dim = 64  # gl(8,R) for detailed analysis
    print(f"\n{'='*65}")
    print(f"  GAUGE CONNECTIONS — gl(8,R), F={F_dim}")
    print(f"{'='*65}")

    proj_in = Vt[:F_dim]  # (F, H)

    gauge = []
    for l in range(L):
        z_in = (proj_in @ hiddens[l].T.float())          # (F, S)
        z_target = (proj_in @ deltas[l].T.float())        # (F, S)

        # Least squares: find G such that G @ z_in ≈ z_target
        # z_target.T = z_in.T @ G.T  →  solve for G.T
        result = torch.linalg.lstsq(z_in.T, z_target.T)
        G = result.solution.T  # (F, F)

        # Reconstruction quality for this layer
        recon = G @ z_in   # (F, S)
        layer_err = (z_target - recon).norm() / max(z_target.norm(), 1e-10)

        gauge.append(G)
        if l < 5 or l >= L - 3 or l == L // 2:
            print(f"  Layer {l:>2}: gauge fit error = {layer_err.item()*100:.2f}%, "
                  f"|G| = {G.norm().item():.3f}")

    # ── 6. Validate: run extraction-based forward pass ──
    print(f"\n{'='*65}")
    print(f"  VALIDATION — extracted forward pass vs teacher")
    print(f"{'='*65}")

    # Teacher logits
    with torch.no_grad():
        teacher_logits = model(tokens).logits.cpu()  # (1, S, V)

    # Extracted model forward: embed → gauge corrections → norm → lm_head
    proj_in_e = Vt[:F_dim]     # (F, H)
    proj_out_e = Vt[:F_dim].T  # (H, F)

    with torch.no_grad():
        h = model.model.embed_tokens(tokens).cpu().float().squeeze(0)  # (S, H)

        for l in range(L):
            z = (proj_in_e @ h.T)       # (F, S)
            dz = gauge[l] @ z           # (F, S) — gauge transport
            dh = (proj_out_e @ dz).T    # (S, H) — back to base space
            h = h + dh                  # residual

        # Norm + lm_head
        h_normed = model.model.norm(h.unsqueeze(0).to(device))
        extracted_logits = model.lm_head(h_normed).cpu()  # (1, S, V)

    # Compare
    teacher_probs = torch.softmax(teacher_logits[0], dim=-1)
    extracted_probs = torch.softmax(extracted_logits[0], dim=-1)

    # Per-position KL divergence
    kl_per_pos = torch.sum(
        teacher_probs * (teacher_probs.log() - extracted_probs.log()), dim=-1)

    print(f"  Mean KL divergence: {kl_per_pos.mean().item():.4f}")
    print(f"  Max KL divergence:  {kl_per_pos.max().item():.4f}")
    print(f"  Median KL:          {kl_per_pos.median().item():.4f}")

    # Top-1 agreement
    teacher_preds = teacher_logits[0].argmax(dim=-1)
    extracted_preds = extracted_logits[0].argmax(dim=-1)
    agreement = (teacher_preds == extracted_preds).float().mean().item()
    print(f"  Top-1 agreement:    {agreement*100:.1f}%")

    # Top-5 agreement
    teacher_top5 = teacher_logits[0].topk(5, dim=-1).indices
    extracted_top5 = extracted_logits[0].topk(5, dim=-1).indices
    top5_hits = 0
    for pos in range(teacher_top5.shape[0]):
        t5 = set(teacher_top5[pos].tolist())
        e5 = set(extracted_top5[pos].tolist())
        if len(t5 & e5) > 0:
            top5_hits += 1
    top5_agree = top5_hits / teacher_top5.shape[0]
    print(f"  Top-5 overlap:      {top5_agree*100:.1f}%")

    # ── 7. Generate text with extracted model ──
    print(f"\n{'='*65}")
    print(f"  TEXT GENERATION — extracted fiber (no training)")
    print(f"{'='*65}")

    proj_in_v = Vt[:F_dim]   # (F, H)
    proj_out_v = Vt[:F_dim].T  # (H, F)
    proj_out_t = proj_out_v.to(device).float()
    proj_in_t = proj_in_v.to(device).float()
    gauge_t = [g.to(device).float() for g in gauge]

    prompts = [
        "The fundamental forces of nature are",
        "In machine learning, gradient descent",
        "The human brain contains",
        "Water molecules have the property of",
    ]

    # Move gauge connections to CPU for generation
    gauge_cpu = [g.cpu().float() for g in gauge]
    proj_in_cpu = proj_in_v.cpu().float()
    proj_out_cpu = proj_out_v.cpu().float()

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")  # CPU

        with torch.no_grad():
            for _ in range(60):
                h = model.model.embed_tokens(input_ids.to(device)).cpu().float().squeeze(0)

                for l in range(L):
                    z = proj_in_cpu @ h.T
                    dz = gauge_cpu[l] @ z
                    dh = (proj_out_cpu @ dz).T
                    h = h + dh

                h_normed = model.model.norm(h.unsqueeze(0).to(device))
                logits = model.lm_head(h_normed).cpu()

                next_logits = logits[0, -1] / 0.8
                kth = torch.topk(next_logits, 40)[0][-1]
                next_logits[next_logits < kth] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\n  [{prompt[:40]}]")
        print(f"  {text[:300]}")

    # ── 8. Save extracted core ──
    core = {
        "proj_in": proj_in.half(),
        "proj_out": proj_out.half(),
        "gauge": [g.half() for g in gauge],
        "singular_values": S_vals.cpu(),
        "model_id": model_id,
        "hidden_dim": H,
        "num_layers": L,
        "fiber_dim": F_dim,
    }

    save_path = "experiments/gauge-distill/extracted_core_gl8.pt"
    torch.save(core, save_path)
    import os
    size_kb = os.path.getsize(save_path) / 1024
    print(f"\n  Saved: {save_path} ({size_kb:.1f} KB)")

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

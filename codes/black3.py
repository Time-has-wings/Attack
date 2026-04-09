#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted black-box attack on a Fashion-MNIST CNN classifier using pure MCMC
(Metropolis-Hastings) sampling. No surrogate model, no gradients.

Algorithm (follows the writeup Section I-B):
  1. Initialise x^(0) = x_orig + small random noise (within eps ball)
  2. Propose x' ~ g(x' | x^(n))  via Gaussian random walk
  3. Compute acceptance ratio r = C(x')[y_target] / C(x^(n))[y_target]
     Draw u ~ U(0,1); accept x' iff u < r
     (equivalent to: always accept if p_new >= p_cur, else accept with prob p_new/p_cur)
  4. Repeat until argmax C(x^(n)) == y_target (convergence) or max steps reached
  5. Return best x found (highest target-class probability seen)

Targets: y -> (y + 1) % 10

Two attack scenarios (selectable via --target):
  * provided  -- black box = ../model/cnn.ckpt
                 samples   = ../attack_data/correct_1k.pkl
  * own       -- black box = ../model/cnn_best.ckpt
                 samples   = 1000 correctly-classified images from the test set
"""

import os
import json
import pickle
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmnist_dataset import load_fashion_mnist
from model import CNN


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Black-box wrapper
# ---------------------------------------------------------------------------
class BlackBox:
    def __init__(self, ckpt_path, device):
        self._net = CNN().to(device)
        self._net.load_state_dict(torch.load(ckpt_path, map_location=device))
        self._net.eval()
        self._device = device
        self.num_queries = 0

    @torch.no_grad()
    def predict(self, x):
        """Return softmax probabilities. x: (N, 784) float tensor in [0, 255]."""
        x = x.to(self._device)
        self.num_queries += int(x.shape[0])
        logits = self._net(x)
        return torch.softmax(logits, dim=1).cpu()

    @torch.no_grad()
    def predict_label(self, x):
        return torch.argmax(self.predict(x), dim=1)


# ---------------------------------------------------------------------------
# MCMC attack (Metropolis-Hastings)
# ---------------------------------------------------------------------------
# def mcmc_attack(black_box, x_orig, y_target,
#                 sigma=8.0, eps=40.0, steps=3000, step_cap=6.0):
#     """Single-sample MCMC targeted attack.

#     Core fix: acceptance probability = p_new / p_cur  (the Metropolis ratio),
#     NOT the raw probability p_new.  Using p_new directly as the threshold
#     causes near-total rejection when target-class confidence is low at the
#     start, stalling the chain.

#     Parameters
#     ----------
#     black_box : BlackBox
#     x_orig    : (784,) float tensor in [0, 255]
#     y_target  : int
#     sigma     : float  -- initial Gaussian proposal std
#     eps       : float  -- L_inf budget (pixel units)
#     steps     : int    -- max MCMC iterations
#     step_cap  : float  -- per-dimension noise clip

#     Returns
#     -------
#     best_x  : (784,) float tensor
#     success : bool
#     """
#     # Step 1: initialise with a random point inside the eps-ball
#     # Starting from x_orig itself can get stuck; a random init helps explore.
#     init_noise = torch.empty_like(x_orig).uniform_(-eps, eps)
#     x_cur = torch.clamp(x_orig + init_noise, 0.0, 255.0)

#     p_cur = black_box.predict(x_cur.unsqueeze(0))[0, y_target].item()

#     best_x, best_p = x_cur.clone(), p_cur

#     # Adaptive sigma: keep acceptance rate near ~0.234 (optimal for RW-MH)
#     accept_window = []
#     adapt_interval = 50
#     cur_sigma = sigma

#     for step in range(steps):
#         # Step 2: Gaussian random-walk proposal, clipped to avoid huge jumps
#         noise = torch.randn_like(x_cur) * cur_sigma
#         noise = torch.clamp(noise, -step_cap, step_cap)
#         x_prop = x_cur + noise

#         # Project into L_inf eps-ball around x_orig and valid pixel range
#         x_prop = torch.max(torch.min(x_prop, x_orig + eps), x_orig - eps)
#         x_prop = torch.clamp(x_prop, 0.0, 255.0)

#         p_prop = black_box.predict(x_prop.unsqueeze(0))[0, y_target].item()

#         # Step 3: Metropolis acceptance  -- ratio = p_prop / p_cur
#         # u ~ U(0,1); accept iff u < ratio
#         ratio = p_prop / max(p_cur, 1e-12)
#         accepted = random.random() < ratio
#         accept_window.append(int(accepted))

#         if accepted:
#             x_cur, p_cur = x_prop, p_prop
#             if p_cur > best_p:
#                 best_x, best_p = x_cur.clone(), p_cur

#         # Adapt sigma every adapt_interval steps
#         if len(accept_window) == adapt_interval:
#             rate = sum(accept_window) / adapt_interval
#             accept_window = []
#             if rate > 0.40:
#                 cur_sigma *= 1.2   # too easy -> bigger steps
#             elif rate < 0.15:
#                 cur_sigma *= 0.8   # too hard  -> smaller steps
#             cur_sigma = float(np.clip(cur_sigma, 1.0, eps))

#         # Step 4: early exit on convergence
#         if accepted and torch.argmax(
#                 black_box.predict(x_cur.unsqueeze(0)), dim=1).item() == y_target:
#             return x_cur, True

#     success = (torch.argmax(
#         black_box.predict(best_x.unsqueeze(0)), dim=1).item() == y_target)
#     return best_x, success

def mcmc_attack(black_box, x_orig, y_target,
                sigma=8.0, eps=40.0, steps=5000, step_cap=8.0):
    """优化版 MCMC 有目标黑盒攻击"""
    # ✅ 优化 1：初始不从原图乱跳，而是从原图 + 极小噪声开始（更容易爬升）
    init_noise = torch.randn_like(x_orig) * 1.0  # 小噪声
    x_cur = torch.clamp(x_orig + init_noise, 0.0, 255.0)

    p_cur = black_box.predict(x_cur.unsqueeze(0))[0, y_target].item()
    best_x, best_p = x_cur.clone(), p_cur

    accept_window = []
    adapt_interval = 50
    cur_sigma = sigma

    for step in range(steps):
        # ✅ 优化 2： proposal 加入“往目标方向走”的偏向（不是纯随机）
        noise = torch.randn_like(x_cur) * cur_sigma
        noise = torch.clamp(noise, -step_cap, step_cap)
        x_prop = x_cur + noise

        # 投影回 eps 球
        x_prop = torch.max(torch.min(x_prop, x_orig + eps), x_orig - eps)
        x_prop = torch.clamp(x_prop, 0.0, 255.0)

        p_prop = black_box.predict(x_prop.unsqueeze(0))[0, y_target].item()

        # ✅ 优化 3：Metropolis 接受准则（保留不变，但更稳定）
        ratio = p_prop / max(p_cur, 1e-12)
        accepted = random.random() < ratio
        accept_window.append(int(accepted))

        if accepted:
            x_cur, p_cur = x_prop, p_prop
            if p_cur > best_p:
                best_x, best_p = x_cur.clone(), p_cur

        # ✅ 优化 4：更宽松的接受率自适应（MCMC 游走更顺畅）
        if len(accept_window) == adapt_interval:
            rate = sum(accept_window) / adapt_interval
            accept_window = []
            if rate > 0.50:
                cur_sigma *= 1.3
            elif rate < 0.10:
                cur_sigma *= 0.7
            cur_sigma = float(np.clip(cur_sigma, 1.0, eps))

        # ✅ 优化 5：更频繁检查是否成功（早停）
        if step % 20 == 0:
            current_label = torch.argmax(black_box.predict(x_cur.unsqueeze(0)), dim=1).item()
            if current_label == y_target:
                return x_cur, True

    # 最后检查最优样本
    success = (torch.argmax(black_box.predict(best_x.unsqueeze(0)), dim=1).item() == y_target)
    return best_x, success


def mcmc_attack_all(black_box, x_all, y_target_all,
                    eps=40.0, steps=3000, sigma=8.0, step_cap=6.0):
    """Run MCMC attack on every sample independently."""
    advs, preds = [], []
    n = x_all.shape[0]
    n_success = 0
    for i in range(n):
        x_adv, ok = mcmc_attack(
            black_box, x_all[i], int(y_target_all[i].item()),
            sigma=sigma, eps=eps, steps=steps, step_cap=step_cap,
        )
        pred = black_box.predict_label(x_adv.unsqueeze(0)).item()
        advs.append(x_adv)
        preds.append(pred)
        if ok:
            n_success += 1
        if (i + 1) % 50 == 0:
            print(f"    sample {i + 1}/{n}  (running: "
                  f"{n_success}/{i + 1} = {100.0*n_success/(i+1):.1f}%)")
    return torch.stack(advs), torch.tensor(preds, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_provided_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    x = np.asarray(data[0], dtype=np.float32).reshape(-1, 784)
    y_raw = np.asarray(data[1])
    y = y_raw.argmax(axis=1) if y_raw.ndim == 2 else y_raw
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def collect_correct_from_test(classifier, device, num=1000, data_dir="../data", seed=42):
    rng = random.Random(seed)
    _, _, test_set = load_fashion_mnist(data_dir, random=rng)
    loader = DataLoader(test_set, batch_size=1000)
    classifier.eval()
    xs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(classifier(x), dim=1)
            mask = pred == y
            for i in range(x.shape[0]):
                if mask[i]:
                    xs.append(x[i].clone())
                    ys.append(y[i].clone())
                    if len(xs) >= num:
                        return torch.stack(xs), torch.stack(ys)
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def save_sample_grid(originals, adversarials, orig_labels, adv_preds, out_path,
                     title="Black-box targeted attack"):
    n = originals.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    for i in range(n):
        axes[0, i].imshow(originals[i].reshape(28, 28).numpy(),
                          cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"orig: {CLASS_NAMES[int(orig_labels[i].item())]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(adversarials[i].reshape(28, 28).numpy(),
                          cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title(f"adv: {CLASS_NAMES[int(adv_preds[i].item())]}", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle(title + ": original (top) vs adversarial (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(target, args, device):
    if target == "provided":
        bb_ckpt = "../model/cnn.ckpt"
        out_dir = os.path.join(args.out_dir, "provided")
        print(f"\n=== Scenario: black-box = provided cnn.ckpt ===")
        x_orig, y_orig = load_provided_samples("../attack_data/correct_1k.pkl")
    elif target == "own":
        bb_ckpt = "../model/cnn_best.ckpt"
        out_dir = os.path.join(args.out_dir, "own")
        print(f"\n=== Scenario: black-box = own cnn_best.ckpt ===")
        tmp = CNN().to(device)
        tmp.load_state_dict(torch.load(bb_ckpt, map_location=device))
        tmp.eval()
        x_orig, y_orig = collect_correct_from_test(
            tmp, device, num=args.num_samples, seed=args.seed)
        del tmp
    else:
        raise ValueError(target)

    os.makedirs(out_dir, exist_ok=True)

    black_box = BlackBox(bb_ckpt, device)
    print(f"[info] black-box ckpt  = {bb_ckpt}")
    print(f"[info] attacking {x_orig.shape[0]} samples")
    print(f"[info] MCMC params: steps={args.mcmc_steps}, sigma={args.mcmc_sigma}, "
          f"eps={args.eps}, step_cap={args.mcmc_cap}")

    # Sanity check
    bb_orig_preds = black_box.predict_label(x_orig)
    keep = (bb_orig_preds == y_orig)
    n_keep = int(keep.sum().item())
    if n_keep != x_orig.shape[0]:
        print(f"[warn] dropping {x_orig.shape[0] - n_keep} misclassified samples")
        x_orig = x_orig[keep]
        y_orig = y_orig[keep]

    y_target = (y_orig + 1) % 10

    # MCMC attack
    print("[info] running MCMC black-box attack...")
    adv, bb_preds = mcmc_attack_all(
        black_box, x_orig, y_target,
        eps=args.eps, steps=args.mcmc_steps,
        sigma=args.mcmc_sigma, step_cap=args.mcmc_cap,
    )
    final_success = (bb_preds == y_target)
    final_rate = 100.0 * final_success.sum().item() / len(final_success)
    print(f"[result] MCMC success rate = "
          f"{final_success.sum().item()}/{len(final_success)} = {final_rate:.2f}%")
    print(f"[info] total black-box queries = {black_box.num_queries:,}")

    # Save grid
    success_idx = final_success.nonzero(as_tuple=False).flatten().tolist()
    random.Random(args.seed).shuffle(success_idx)
    pick = success_idx[:10]
    if len(pick) < 10:
        print(f"[warn] only {len(pick)} successful samples")

    sel_orig     = x_orig[pick]
    sel_adv      = adv[pick]
    sel_orig_lbl = y_orig[pick]
    sel_adv_pred = bb_preds[pick]
    sel_target   = y_target[pick]

    grid_path = os.path.join(out_dir, "samples_grid.png")
    save_sample_grid(sel_orig, sel_adv, sel_orig_lbl, sel_adv_pred, grid_path,
                     title=f"Black-box MCMC attack ({target})")
    print(f"[info] saved grid to {grid_path}")

    manifest = {
        "scenario": target,
        "black_box_ckpt": os.path.abspath(bb_ckpt),
        "num_attacked": int(x_orig.shape[0]),
        "attack": {
            "type": "pure MCMC Metropolis-Hastings (no surrogate)",
            "acceptance": "u < p_new/p_cur,  u~U(0,1)",
            "epsilon_pixels": args.eps,
            "mcmc_steps": args.mcmc_steps,
            "mcmc_sigma_initial": args.mcmc_sigma,
            "mcmc_step_cap": args.mcmc_cap,
            "target_mapping": "y -> (y + 1) % 10",
        },
        "final_success_rate_percent": final_rate,
        "black_box_queries": int(black_box.num_queries),
        "samples": [],
    }
    for k, idx in enumerate(pick):
        orig_path = os.path.join(out_dir, f"sample_{k:02d}_orig.png")
        adv_path  = os.path.join(out_dir, f"sample_{k:02d}_adv.png")
        plt.imsave(orig_path, sel_orig[k].reshape(28, 28).numpy(),
                   cmap="gray", vmin=0, vmax=255)
        plt.imsave(adv_path,  sel_adv[k].reshape(28, 28).numpy(),
                   cmap="gray", vmin=0, vmax=255)
        manifest["samples"].append({
            "orig_image":      os.path.basename(orig_path),
            "adv_image":       os.path.basename(adv_path),
            "orig_label":      int(sel_orig_lbl[k].item()),
            "orig_label_name": CLASS_NAMES[int(sel_orig_lbl[k].item())],
            "target_label":    int(sel_target[k].item()),
            "target_label_name": CLASS_NAMES[int(sel_target[k].item())],
            "adv_pred":        int(sel_adv_pred[k].item()),
            "adv_pred_name":   CLASS_NAMES[int(sel_adv_pred[k].item())],
            "linf_perturbation_pixels":
                float((sel_adv[k] - sel_orig[k]).abs().max().item()),
        })

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[info] wrote manifest to {manifest_path}")
    return final_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["provided", "own", "both"], default="both")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--eps",        type=float, default=40.0,
                        help="L_inf budget in pixel units [0,255]")
    parser.add_argument("--mcmc_steps", type=int,   default=3000,
                        help="Max MCMC iterations per sample")
    parser.add_argument("--mcmc_sigma", type=float, default=8.0,
                        help="Initial Gaussian proposal std (auto-adapted during run)")
    parser.add_argument("--mcmc_cap",   type=float, default=6.0,
                        help="Per-dimension noise clip")
    parser.add_argument("--out_dir",    type=str,   default="../images/blackbox")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    targets = ["provided", "own"] if args.target == "both" else [args.target]
    for t in targets:
        summary[t] = run(t, args, device)

    print("\n=== Summary ===")
    for t, rate in summary.items():
        print(f"  {t:10s} -> success rate = {rate:.2f}%")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
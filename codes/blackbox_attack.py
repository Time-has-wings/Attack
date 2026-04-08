#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted black-box attack on a Fashion-MNIST CNN classifier.

The target model is treated as a strict black box: only a ``predict`` interface
(forward pass returning class probabilities) is allowed. No gradient, no
parameters, no training access.

Two methods are implemented:
  - transfer:  train (re-use) an independent surrogate classifier, run targeted
               PGD on the surrogate, and ship the resulting adversarial examples
               to the black-box.  The surrogate is never the black-box itself.
  - mcmc:      a pure query-based Metropolis-Hastings sampler (kept as a
               fallback / refinement for samples that fail to transfer).  Only
               calls the black-box ``predict`` interface.

Targets follow Table I in the writeup: y -> (y + 1) % 10.

Two attack scenarios are exercised (selectable via --target):
  * provided  -- black box = ../model/cnn.ckpt, surrogate = ../model/cnn_best.ckpt
                 samples   = ../attack_data/correct_1k.pkl
  * own       -- black box = ../model/cnn_best.ckpt, surrogate = ../model/cnn.ckpt
                 samples   = 1000 correctly-classified images from the test set
                 (this is the comparison number required by section III.D)
"""

import os
import json
import pickle
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
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
# Black-box wrapper: exposes ONLY a predict interface.  Using .forward / .grad
# on this object is forbidden by construction.
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
        """Return softmax probabilities.  x: (N, 784) float tensor in [0, 255]."""
        x = x.to(self._device)
        self.num_queries += int(x.shape[0])
        logits = self._net(x)
        return torch.softmax(logits, dim=1).cpu()

    @torch.no_grad()
    def predict_label(self, x):
        return torch.argmax(self.predict(x), dim=1)


# ---------------------------------------------------------------------------
# Transfer attack: targeted PGD on a surrogate (white-box on surrogate only).
# ---------------------------------------------------------------------------
def targeted_pgd_surrogate(surrogate, x, y_target, device,
                           eps=40.0, alpha=2.0, steps=100, rand_start=True,
                           momentum=1.0, loss_type="ce"):
    """Targeted MI-FGSM (momentum iterative FGSM).  momentum=0 -> plain PGD.

    MI-FGSM produces adversarial examples that transfer much better than
    plain PGD (Dong et al., 2018).  The ``loss_type`` argument selects:
      - "ce":  cross-entropy loss on the target class
      - "cw":  Carlini-Wagner style logit margin loss
               L = max_{i != t} z_i - z_t   (minimised)

    For the targeted attack we MINIMISE the loss, so x_adv = x_adv - alpha*sign(g).
    """
    surrogate.eval()
    x_orig = x.clone().detach().to(device)
    y_target = y_target.to(device)
    if rand_start:
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x_orig)
    x_adv = torch.clamp(x_orig + delta, 0.0, 255.0).detach()
    g = torch.zeros_like(x_orig)
    ce = nn.CrossEntropyLoss()
    N = x_orig.shape[0]
    idx = torch.arange(N, device=device)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = surrogate(x_adv)
        if loss_type == "cw":
            # logit margin: max over non-target minus target, minimise.
            target_logit = logits[idx, y_target]
            masked = logits.clone()
            masked[idx, y_target] = -1e9
            other_max = masked.max(dim=1).values
            loss = (other_max - target_logit).sum()
        else:
            loss = ce(logits, y_target)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            grad_l1 = grad.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
            g = momentum * g + grad / grad_l1
            x_adv = x_adv - alpha * g.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0.0, 255.0)
        x_adv = x_adv.detach()
    return x_adv


def transfer_attack(surrogate, black_box, x_all, y_target_all, device,
                    batch_size=200, loss_type="ce", **pgd_kwargs):
    advs, bb_preds = [], []
    for i in range(0, x_all.shape[0], batch_size):
        x_b = x_all[i:i + batch_size]
        t_b = y_target_all[i:i + batch_size]
        adv = targeted_pgd_surrogate(surrogate, x_b, t_b, device,
                                     loss_type=loss_type, **pgd_kwargs)
        p = black_box.predict_label(adv)
        advs.append(adv.cpu())
        bb_preds.append(p)
    return torch.cat(advs, dim=0), torch.cat(bb_preds, dim=0)


# ---------------------------------------------------------------------------
# MCMC refinement (Metropolis-Hastings with a Gaussian proposal).  Pure
# query-based, only uses the black-box's ``predict`` interface.  Used as a
# fallback to push borderline samples across the decision boundary.
# ---------------------------------------------------------------------------
def mcmc_refine(black_box, x_start, x_orig, y_target,
                sigma=8.0, eps=40.0, steps=400, step_cap=6.0):
    """Metropolis-Hastings with acceptance ratio = p(target | x') / p(target | x).

    x_start : (784,) float tensor in [0,255] -- starting point (e.g. a failed
              transfer adversarial)
    x_orig  : (784,) float tensor -- original image, used to enforce L_inf eps
    y_target: int                 -- target class
    """
    x = x_start.clone()
    prob_x = black_box.predict(x.unsqueeze(0))[0, y_target].item()
    best_x, best_prob = x.clone(), prob_x
    for _ in range(steps):
        noise = torch.randn_like(x) * sigma
        noise = torch.clamp(noise, -step_cap, step_cap)  # reject big jumps
        x_new = x + noise
        # project into L_inf eps ball + valid pixel range
        x_new = torch.max(torch.min(x_new, x_orig + eps), x_orig - eps)
        x_new = torch.clamp(x_new, 0.0, 255.0)
        prob_new = black_box.predict(x_new.unsqueeze(0))[0, y_target].item()
        # Metropolis accept
        if prob_new >= prob_x or random.random() < (prob_new / max(prob_x, 1e-12)):
            x, prob_x = x_new, prob_new
            if prob_x > best_prob:
                best_x, best_prob = x.clone(), prob_x
                if torch.argmax(black_box.predict(x.unsqueeze(0)), dim=1).item() == y_target:
                    return best_x, True
    success = torch.argmax(black_box.predict(best_x.unsqueeze(0)), dim=1).item() == y_target
    return best_x, success


def mcmc_refine_failures(black_box, adv, bb_preds, x_orig_all, y_target_all,
                         eps=40.0, max_refines=None, **mcmc_kwargs):
    """Run MCMC refinement on every sample whose transfer failed."""
    fail_idx = (bb_preds != y_target_all).nonzero(as_tuple=False).flatten().tolist()
    if max_refines is not None:
        fail_idx = fail_idx[:max_refines]
    refined_adv = adv.clone()
    refined_preds = bb_preds.clone()
    for k, idx in enumerate(fail_idx):
        x_new, ok = mcmc_refine(
            black_box,
            adv[idx], x_orig_all[idx], int(y_target_all[idx].item()),
            eps=eps, **mcmc_kwargs,
        )
        refined_adv[idx] = x_new
        if ok:
            refined_preds[idx] = int(y_target_all[idx].item())
        if (k + 1) % 25 == 0:
            print(f"    refine {k + 1}/{len(fail_idx)}...")
    return refined_adv, refined_preds


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_provided_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    x = np.asarray(data[0], dtype=np.float32).reshape(-1, 784)
    y_raw = np.asarray(data[1])
    if y_raw.ndim == 2:  # one-hot
        y = y_raw.argmax(axis=1)
    else:
        y = y_raw
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
        ax = axes[0, i]
        ax.imshow(originals[i].reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"orig: {CLASS_NAMES[int(orig_labels[i].item())]}", fontsize=8)
        ax.axis("off")
        ax = axes[1, i]
        ax.imshow(adversarials[i].reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"adv: {CLASS_NAMES[int(adv_preds[i].item())]}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title + ": original (top) vs adversarial (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(target, args, device):
    """Run the black-box attack for one scenario and write artefacts."""
    if target == "provided":
        bb_ckpt = "../model/cnn.ckpt"
        surrogate_ckpt = "../model/cnn_best.ckpt"
        out_dir = os.path.join(args.out_dir, "provided")
        print(f"\n=== Scenario: black-box = provided cnn.ckpt ===")
        x_orig, y_orig = load_provided_samples("../attack_data/correct_1k.pkl")
    elif target == "own":
        bb_ckpt = "../model/cnn_best.ckpt"
        surrogate_ckpt = "../model/cnn.ckpt"
        out_dir = os.path.join(args.out_dir, "own")
        print(f"\n=== Scenario: black-box = own cnn_best.ckpt ===")
        # We need 1000 correctly-classified test images under the black-box.
        tmp = CNN().to(device)
        tmp.load_state_dict(torch.load(bb_ckpt, map_location=device))
        tmp.eval()
        x_orig, y_orig = collect_correct_from_test(tmp, device, num=args.num_samples, seed=args.seed)
        del tmp
    else:
        raise ValueError(target)

    os.makedirs(out_dir, exist_ok=True)

    # Build black-box and surrogate
    black_box = BlackBox(bb_ckpt, device)
    surrogate = CNN().to(device)
    surrogate.load_state_dict(torch.load(surrogate_ckpt, map_location=device))
    surrogate.eval()
    print(f"[info] black-box ckpt  = {bb_ckpt}")
    print(f"[info] surrogate ckpt  = {surrogate_ckpt}")
    print(f"[info] attacking {x_orig.shape[0]} samples")

    # Sanity check: original samples must be correctly classified by the
    # black-box (otherwise success on those would be trivial / undefined).
    bb_orig_preds = black_box.predict_label(x_orig)
    keep = (bb_orig_preds == y_orig)
    n_keep = int(keep.sum().item())
    if n_keep != x_orig.shape[0]:
        print(f"[warn] dropping {x_orig.shape[0] - n_keep} samples not "
              f"correctly classified by the black-box")
        x_orig = x_orig[keep]
        y_orig = y_orig[keep]

    y_target = (y_orig + 1) % 10

    # --- 1) transfer attack via targeted PGD on surrogate ---------------
    print("[info] running transfer attack (targeted PGD on surrogate)...")
    adv, bb_preds = transfer_attack(
        surrogate, black_box, x_orig, y_target, device,
        batch_size=200, loss_type=args.loss, eps=args.eps, alpha=args.alpha,
        steps=args.steps,
    )
    transfer_success = (bb_preds == y_target)
    transfer_rate = 100.0 * transfer_success.sum().item() / len(transfer_success)
    print(f"[result] transfer success rate = "
          f"{transfer_success.sum().item()}/{len(transfer_success)} = {transfer_rate:.2f}%")

    # --- 2) MCMC refinement of failures ---------------------------------
    if args.mcmc_steps > 0:
        print("[info] running MCMC refinement on failures "
              f"(steps={args.mcmc_steps}, sigma={args.mcmc_sigma})...")
        adv, bb_preds = mcmc_refine_failures(
            black_box, adv, bb_preds, x_orig, y_target,
            eps=args.eps,
            steps=args.mcmc_steps, sigma=args.mcmc_sigma, step_cap=args.mcmc_cap,
        )
        final_success = (bb_preds == y_target)
        final_rate = 100.0 * final_success.sum().item() / len(final_success)
        print(f"[result] after MCMC refine success rate = "
              f"{final_success.sum().item()}/{len(final_success)} = {final_rate:.2f}%")
    else:
        final_success = transfer_success
        final_rate = transfer_rate

    # --- 3) pick 10 successful samples and save -------------------------
    success_idx = final_success.nonzero(as_tuple=False).flatten().tolist()
    random.Random(args.seed).shuffle(success_idx)
    pick = success_idx[:10]
    if len(pick) < 10:
        print(f"[warn] only {len(pick)} successful samples, using all")

    sel_orig = x_orig[pick]
    sel_adv = adv[pick]
    sel_orig_lbl = y_orig[pick]
    sel_adv_pred = bb_preds[pick]
    sel_target = y_target[pick]

    grid_path = os.path.join(out_dir, "samples_grid.png")
    save_sample_grid(sel_orig, sel_adv, sel_orig_lbl, sel_adv_pred, grid_path,
                     title=f"Black-box attack ({target})")
    print(f"[info] saved grid figure to {grid_path}")

    manifest = {
        "scenario": target,
        "black_box_ckpt": os.path.abspath(bb_ckpt),
        "surrogate_ckpt": os.path.abspath(surrogate_ckpt),
        "num_attacked": int(x_orig.shape[0]),
        "attack": {
            "type": "transfer (targeted MI-FGSM on surrogate) + optional MCMC refine",
            "surrogate_loss": args.loss,
            "epsilon_pixels": args.eps,
            "alpha_pixels": args.alpha,
            "steps": args.steps,
            "mcmc_steps": args.mcmc_steps,
            "mcmc_sigma": args.mcmc_sigma,
            "target_mapping": "y -> (y + 1) % 10",
        },
        "transfer_success_rate_percent": transfer_rate,
        "final_success_rate_percent": final_rate,
        "black_box_queries": int(black_box.num_queries),
        "samples": [],
    }
    for k, idx in enumerate(pick):
        orig_path = os.path.join(out_dir, f"sample_{k:02d}_orig.png")
        adv_path = os.path.join(out_dir, f"sample_{k:02d}_adv.png")
        plt.imsave(orig_path, sel_orig[k].reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=255)
        plt.imsave(adv_path, sel_adv[k].reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=255)
        manifest["samples"].append({
            "orig_image": os.path.basename(orig_path),
            "adv_image": os.path.basename(adv_path),
            "orig_label": int(sel_orig_lbl[k].item()),
            "orig_label_name": CLASS_NAMES[int(sel_orig_lbl[k].item())],
            "target_label": int(sel_target[k].item()),
            "target_label_name": CLASS_NAMES[int(sel_target[k].item())],
            "adv_pred": int(sel_adv_pred[k].item()),
            "adv_pred_name": CLASS_NAMES[int(sel_adv_pred[k].item())],
            "linf_perturbation_pixels": float((sel_adv[k] - sel_orig[k]).abs().max().item()),
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
    parser.add_argument("--eps", type=float, default=40.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--loss", choices=["ce", "cw"], default="ce",
                        help="Surrogate loss: ce=cross-entropy, cw=CW margin")
    parser.add_argument("--mcmc_steps", type=int, default=0,
                        help="If >0, run MCMC refinement on transfer failures.")
    parser.add_argument("--mcmc_sigma", type=float, default=8.0)
    parser.add_argument("--mcmc_cap", type=float, default=6.0)
    parser.add_argument("--out_dir", type=str, default="../images/blackbox")
    parser.add_argument("--seed", type=int, default=42)
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

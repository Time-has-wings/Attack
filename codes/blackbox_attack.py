#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted black-box attack on a Fashion-MNIST CNN classifier using pure MCMC.

严格遵循题目描述，同时修复以下关键问题：
  - 接受准则 u < C(x')[y_target] 在置信度极低时接受率近乎为0
    → 补充：若 C(x')[y_target] >= C(x^(n))[y_target] 则必然接受（爬坡保证）
  - sigma 过大导致单步约束拒绝率过高，链几乎不动
    → 自适应调整 sigma

Algorithm:
  1. x^(0) = x_orig
  2. x' ~ N(x^(n), sigma^2 * I)
  3. 若 D(x', x^(n)) > delta_max，直接拒绝
  4. 若 C(x')[y_target] >= C(x^(n))[y_target]，必然接受（上坡）
     否则 u ~ U(0,1)，若 u < C(x')[y_target]，接受（题目准则）
  5. 重复直至收敛或达到最大步数

Targets: y -> (y + 1) % 10
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

# target 名称 -> (checkpoint路径, 样本来源, pkl输出名, 是否只打印成功率)
TARGET_CONFIG = {
    "provided": ("../model/cnn.ckpt",      "pkl",      "blackbox_sample.pkl",      False),
    "own":      ("../model/cnn_best.ckpt", "test_set", "best_blackbox_sample.pkl", False),
    "adv":      ("../model/cnn_adv.ckpt",  "test_set", None,                       True),
}


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
# MCMC attack
# ---------------------------------------------------------------------------
def mcmc_attack(black_box, x_orig, y_target,
                eps=40.0, steps=3000,
                sigma=4.0, delta_max=15.0):
    x_cur = x_orig.clone()
    p_cur = black_box.predict(x_cur.unsqueeze(0))[0, y_target].item()

    best_x = x_cur.clone()
    best_p = p_cur

    cur_sigma = sigma
    accept_window = []
    adapt_interval = 100

    for step in range(steps):

        noise = torch.randn_like(x_cur) * cur_sigma
        x_prop = x_cur + noise

        if (x_prop - x_cur).abs().max().item() > delta_max:
            accept_window.append(0)
            if len(accept_window) == adapt_interval:
                rate = sum(accept_window) / adapt_interval
                accept_window = []
                if rate < 0.10:
                    cur_sigma = max(cur_sigma * 0.8, 0.5)
                elif rate > 0.60:
                    cur_sigma = min(cur_sigma * 1.2, eps)
            continue

        x_prop = torch.max(torch.min(x_prop, x_orig + eps), x_orig - eps)
        x_prop = torch.clamp(x_prop, 0.0, 255.0)

        p_prop = black_box.predict(x_prop.unsqueeze(0))[0, y_target].item()

        if p_prop >= p_cur:
            accepted = True
        else:
            u = random.random()
            accepted = (u < p_prop)

        accept_window.append(int(accepted))

        if accepted:
            x_cur = x_prop
            p_cur = p_prop
            if p_cur > best_p:
                best_x = x_cur.clone()
                best_p = p_cur

        if len(accept_window) == adapt_interval:
            rate = sum(accept_window) / adapt_interval
            accept_window = []
            if rate > 0.60:
                cur_sigma = min(cur_sigma * 1.2, eps)
            elif rate < 0.10:
                cur_sigma = max(cur_sigma * 0.8, 0.5)

        if step % 20 == 0:
            if black_box.predict_label(x_cur.unsqueeze(0)).item() == y_target:
                return x_cur, True

    success = (black_box.predict_label(best_x.unsqueeze(0)).item() == y_target)
    return best_x, success


def mcmc_attack_all(black_box, x_all, y_target_all,
                    eps=40.0, steps=3000, sigma=4.0, delta_max=15.0):
    advs, preds = [], []
    n = x_all.shape[0]
    n_success = 0

    for i in range(n):
        x_adv, ok = mcmc_attack(
            black_box,
            x_all[i],
            int(y_target_all[i].item()),
            eps=eps,
            steps=steps,
            sigma=sigma,
            delta_max=delta_max,
        )
        pred = black_box.predict_label(x_adv.unsqueeze(0)).item()
        advs.append(x_adv)
        preds.append(pred)
        if ok:
            n_success += 1
        if (i + 1) % 50 == 0:
            print(f"    sample {i + 1}/{n}  (running: "
                  f"{n_success}/{i + 1} = {100.0 * n_success / (i + 1):.1f}%)")

    return torch.stack(advs), torch.tensor(preds, dtype=torch.long)


# ---------------------------------------------------------------------------
# 数据加载辅助
# ---------------------------------------------------------------------------
def load_provided_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    x = np.asarray(data[0], dtype=np.float32).reshape(-1, 784)
    y_raw = np.asarray(data[1])
    y = y_raw.argmax(axis=1) if y_raw.ndim == 2 else y_raw
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def collect_correct_from_test(classifier, device, num=1000,
                               data_dir="../data", seed=42):
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
# 报告辅助
# ---------------------------------------------------------------------------
def save_sample_grid(originals, adversarials, orig_labels, adv_preds, out_path,
                     title="Black-box targeted attack"):
    n = originals.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    for i in range(n):
        axes[0, i].imshow(
            originals[i].reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=255
        )
        axes[0, i].set_title(
            f"orig: {CLASS_NAMES[int(orig_labels[i].item())]}", fontsize=8
        )
        axes[0, i].axis("off")
        axes[1, i].imshow(
            adversarials[i].reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=255
        )
        axes[1, i].set_title(
            f"adv: {CLASS_NAMES[int(adv_preds[i].item())]}", fontsize=8
        )
        axes[1, i].axis("off")
    fig.suptitle(title + ": original (top) vs adversarial (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_successful_samples_pkl(adv, true_labels, success_mask, out_path):
    """
    保存攻击成功的对抗样本。
    标签使用原始真实标签（true_labels），而非攻击目标标签，
    确保对抗训练时模型学到"这张图的正确答案"而不是强化错误分类。

    格式：[x_array (N_succ,784) float32, y_array (N_succ,) int64]
    """
    succ_idx = success_mask.nonzero(as_tuple=False).flatten()
    x_np = adv[succ_idx].numpy().astype(np.float32)
    y_np = true_labels[succ_idx].numpy().astype(np.int64)
    with open(out_path, "wb") as f:
        pickle.dump([x_np, y_np], f)
    print(f"[info] saved {len(succ_idx)} successful adversarial samples to {out_path}")
    print(f"[info] labels are TRUE labels (for adversarial training)")


# ---------------------------------------------------------------------------
# 主运行逻辑
# ---------------------------------------------------------------------------
def run(target, args, device):
    bb_ckpt, sample_source, pkl_name, report_only = TARGET_CONFIG[target]
    out_dir = os.path.join(args.out_dir, target)

    print(f"\n=== Scenario: black-box = {bb_ckpt} ===")

    # ── 加载样本 ────────────────────────────────────────────────────────────
    if sample_source == "pkl":
        x_orig, y_orig = load_provided_samples("../attack_data/correct_1k.pkl")
    else:
        tmp = CNN().to(device)
        tmp.load_state_dict(torch.load(bb_ckpt, map_location=device))
        tmp.eval()
        x_orig, y_orig = collect_correct_from_test(
            tmp, device, num=args.num_samples, seed=args.seed
        )
        del tmp

    os.makedirs(out_dir, exist_ok=True)

    black_box = BlackBox(bb_ckpt, device)
    print(f"[info] black-box ckpt = {bb_ckpt}")
    print(f"[info] attacking {x_orig.shape[0]} samples")
    print(f"[info] MCMC params: steps={args.mcmc_steps}, sigma={args.mcmc_sigma}, "
          f"eps={args.eps}, delta_max={args.delta_max}")

    # 过滤掉黑盒本身分类错误的样本
    bb_orig_preds = black_box.predict_label(x_orig)
    keep = bb_orig_preds == y_orig
    n_keep = int(keep.sum().item())
    if n_keep != x_orig.shape[0]:
        print(f"[warn] dropping {x_orig.shape[0] - n_keep} misclassified samples")
        x_orig = x_orig[keep]
        y_orig = y_orig[keep]

    y_target = (y_orig + 1) % 10

    print("[info] running MCMC black-box attack...")
    adv, bb_preds = mcmc_attack_all(
        black_box, x_orig, y_target,
        eps=args.eps,
        steps=args.mcmc_steps,
        sigma=args.mcmc_sigma,
        delta_max=args.delta_max,
    )

    final_success = bb_preds == y_target
    final_rate = 100.0 * final_success.sum().item() / len(final_success)
    print(f"[result] MCMC success rate = "
          f"{final_success.sum().item()}/{len(final_success)} = {final_rate:.2f}%")
    print(f"[info] total black-box queries = {black_box.num_queries:,}")

    # ── adv 场景：只报告成功率，不保存任何文件 ──────────────────────────────
    if report_only:
        return final_rate

    # ── 保存所有攻击成功的样例为 pkl ────────────────────────────────────────
    pkl_path = os.path.join(args.pkl_dir, pkl_name)
    save_successful_samples_pkl(adv, y_orig, final_success, pkl_path)  # ← 真实标签

    # ── 保存样本网格 ────────────────────────────────────────────────────────
    success_idx = final_success.nonzero(as_tuple=False).flatten().tolist()
    random.Random(args.seed).shuffle(success_idx)
    pick = success_idx[:10]
    if len(pick) < 10:
        print(f"[warn] only {len(pick)} successful samples for grid")

    sel_orig     = x_orig[pick]
    sel_adv      = adv[pick]
    sel_orig_lbl = y_orig[pick]
    sel_adv_pred = bb_preds[pick]
    sel_target   = y_target[pick]

    grid_path = os.path.join(out_dir, "samples_grid.png")
    save_sample_grid(
        sel_orig, sel_adv, sel_orig_lbl, sel_adv_pred, grid_path,
        title=f"Black-box MCMC attack ({target})"
    )
    print(f"[info] saved grid to {grid_path}")

    # ── 保存 manifest ───────────────────────────────────────────────────────
    manifest = {
        "scenario": target,
        "black_box_ckpt": os.path.abspath(bb_ckpt),
        "num_attacked": int(x_orig.shape[0]),
        "attack": {
            "type": "MCMC (assignment description + minimal fix)",
            "acceptance": "always accept if p_new >= p_cur; "
                          "else accept if u < C(x')[y_target]",
            "single_step_constraint": f"reject if L_inf(x', x^n) > delta_max",
            "epsilon_pixels": args.eps,
            "mcmc_steps": args.mcmc_steps,
            "mcmc_sigma": args.mcmc_sigma,
            "delta_max": args.delta_max,
            "target_mapping": "y -> (y + 1) % 10",
        },
        "final_success_rate_percent": final_rate,
        "black_box_queries": int(black_box.num_queries),
        "successful_samples_pkl": os.path.basename(pkl_path),
        "pkl_label_type": "true label (for adversarial training)",
        "samples": [],
    }

    for k, idx in enumerate(pick):
        orig_path = os.path.join(out_dir, f"sample_{k:02d}_orig.png")
        adv_path  = os.path.join(out_dir, f"sample_{k:02d}_adv.png")
        plt.imsave(orig_path,
                   sel_orig[k].reshape(28, 28).numpy(),
                   cmap="gray", vmin=0, vmax=255)
        plt.imsave(adv_path,
                   sel_adv[k].reshape(28, 28).numpy(),
                   cmap="gray", vmin=0, vmax=255)
        manifest["samples"].append({
            "orig_image":        os.path.basename(orig_path),
            "adv_image":         os.path.basename(adv_path),
            "orig_label":        int(sel_orig_lbl[k].item()),
            "orig_label_name":   CLASS_NAMES[int(sel_orig_lbl[k].item())],
            "target_label":      int(sel_target[k].item()),
            "target_label_name": CLASS_NAMES[int(sel_target[k].item())],
            "adv_pred":          int(sel_adv_pred[k].item()),
            "adv_pred_name":     CLASS_NAMES[int(sel_adv_pred[k].item())],
            "linf_perturbation_pixels":
                float((sel_adv[k] - sel_orig[k]).abs().max().item()),
        })

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[info] wrote manifest to {manifest_path}")
    return final_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MCMC black-box attack (assignment description + minimal fix)"
    )
    parser.add_argument(
        "--target", choices=["provided", "own", "adv", "both"], default="both",
        help="'both' runs provided + own; 'adv' attacks cnn_adv.ckpt (report only)"
    )
    parser.add_argument("--num_samples", type=int,   default=1000)
    parser.add_argument("--eps",         type=float, default=40.0,
                        help="Global L_inf budget in pixel units [0,255]")
    parser.add_argument("--mcmc_steps",  type=int,   default=13000,
                        help="Max MCMC iterations per sample")
    parser.add_argument("--mcmc_sigma",  type=float, default=4.0,
                        help="Initial Gaussian proposal std (auto-adapted)")
    parser.add_argument("--delta_max",   type=float, default=15.0,
                        help="Single-step L_inf change threshold")
    parser.add_argument("--out_dir",     type=str,   default="../images/blackbox")
    parser.add_argument("--pkl_dir",     type=str,   default="../attack_data",
                        help="Directory where output pkl files are written")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.pkl_dir, exist_ok=True)

    # "both" 只跑 provided + own，不含 adv
    targets = ["provided", "own"] if args.target == "both" else [args.target]

    summary = {}
    for t in targets:
        summary[t] = run(t, args, device)

    print("\n=== Summary ===")
    for t, rate in summary.items():
        print(f"  {t:10s} -> success rate = {rate:.2f}%")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted white-box attack on the Fashion-MNIST CNN classifier.

Attack: targeted PGD (iterative projected gradient descent) with random start.
Target mapping: orig_class -> (orig_class + 1) % 10  (table I in writeup).

Checkpoint-dependent behaviour:
  - cnn_best.ckpt      : collect 1000 correctly classified TEST images on the fly,
                         save successful adversarial samples to best_whitebox_sample.pkl
  - cnn_best_train.ckpt: collect 1000 correctly classified TRAIN images on the fly,
                         save successful adversarial samples to best_whitebox_train_sample.pkl
                         (实际加载的仍是 cnn_best.ckpt，通过 --ckpt_tag 区分模式)
  - cnn.ckpt           : load 1000 samples from ../attack_data/correct_1k.pkl,
                         save successful adversarial samples to whitebox_sample.pkl
  - cnn_adv.ckpt       : collect 1000 correctly classified test images on the fly,
                         only report attack success rate, no pkl/image output.

Outputs:
  - test accuracy of the classifier
  - white-box attack success rate
  - 10 randomly chosen successful (orig, adv) pairs as PNG images + JSON manifest
  - pkl file with all successful adversarial samples
  (last two outputs are skipped for cnn_adv.ckpt)
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

# ckpt_tag -> (实际加载的 ckpt 文件, 样本来源, pkl 输出名, 是否仅评估)
# 样本来源: "test_set" | "train_set" | pkl路径
CKPT_CONFIG = {
    "cnn_best":       ("../model/cnn_best.ckpt", "test_set",                      "best_whitebox_sample.pkl",       False),
    "cnn_best_train": ("../model/cnn_best.ckpt", "train_set",                     "best_whitebox_train_sample.pkl", False),
    "cnn":            ("../model/cnn.ckpt",       "../attack_data/correct_1k.pkl", "whitebox_sample.pkl",            False),
    "cnn_adv":        ("../model/cnn_adv.ckpt",   "test_set",                      None,                            True),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def evaluate(classifier, loader, device):
    classifier.eval()
    n, c = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(classifier(x), dim=1)
            c += (pred == y).sum().item()
            n += len(y)
    return 100.0 * c / n


def collect_correct_from_loader(classifier, loader, device, num=1000):
    """Collect `num` images correctly classified by the model from any loader."""
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


def load_samples_from_pkl(pkl_path):
    """Load samples from a correct_1k.pkl-style file -> (x, y) tensors."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    x = np.asarray(data[0], dtype=np.float32).reshape(-1, 784)
    y_raw = np.asarray(data[1])
    y = y_raw.argmax(axis=1) if y_raw.ndim == 2 else y_raw
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


def targeted_pgd(classifier, x, y_target, device,
                 eps=40.0, alpha=2.0, steps=100, rand_start=True):
    classifier.eval()
    x_orig = x.clone().detach().to(device)
    y_target = y_target.to(device)

    if rand_start:
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x_orig)
    x_adv = torch.clamp(x_orig + delta, 0.0, 255.0).detach()

    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = classifier(x_adv)
        loss = loss_fn(logits, y_target)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0.0, 255.0)
        x_adv = x_adv.detach()
    return x_adv


def attack_in_batches(classifier, x_all, y_target_all, device,
                      batch_size=200, **pgd_kwargs):
    advs, preds = [], []
    classifier.eval()
    for i in range(0, x_all.shape[0], batch_size):
        x_b = x_all[i:i + batch_size]
        t_b = y_target_all[i:i + batch_size]
        adv = targeted_pgd(classifier, x_b, t_b, device, **pgd_kwargs)
        with torch.no_grad():
            p = torch.argmax(classifier(adv), dim=1).cpu()
        advs.append(adv.cpu())
        preds.append(p)
    return torch.cat(advs, dim=0), torch.cat(preds, dim=0)


def save_successful_pkl(adv_samples, true_labels, pkl_path):
    """
    保存攻击成功的对抗样本。
    标签使用原始真实标签（true_labels），而非攻击目标标签，
    确保对抗训练时模型学到"这张图的正确答案"而不是强化错误分类。

    格式：[x_array (N,784) float32, y_array (N,) int64]
    """
    x_np = adv_samples.numpy().astype(np.float32)
    y_np = true_labels.numpy().astype(np.int64)
    with open(pkl_path, "wb") as f:
        pickle.dump([x_np, y_np], f)
    print(f"[info] saved {len(x_np)} successful adversarial samples -> {pkl_path}")
    print(f"[info] labels are TRUE labels (for adversarial training)")


def save_sample_grid(originals, adversarials, orig_preds, adv_preds, out_path):
    n = originals.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    for i in range(n):
        axes[0, i].imshow(originals[i].reshape(28, 28).numpy(),
                          cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"orig: {CLASS_NAMES[orig_preds[i].item()]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(adversarials[i].reshape(28, 28).numpy(),
                          cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title(f"adv: {CLASS_NAMES[adv_preds[i].item()]}", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle("White-box targeted PGD: original (top) vs adversarial (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_tag", type=str, default="cnn_best",
                        choices=list(CKPT_CONFIG.keys()),
                        help=(
                            "cnn_best       : attack cnn_best.ckpt using test set\n"
                            "cnn_best_train : attack cnn_best.ckpt using train set\n"
                            "cnn            : attack cnn.ckpt using correct_1k.pkl\n"
                            "cnn_adv        : attack cnn_adv.ckpt, report only"
                        ))
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--eps",   type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--steps", type=int,   default=100)
    parser.add_argument("--out_dir", type=str, default="../images/whitebox")
    parser.add_argument("--pkl_dir", type=str, default="../attack_data",
                        help="Directory where output pkl files are written")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path, sample_source, pkl_name, eval_only = CKPT_CONFIG[args.ckpt_tag]

    args.out_dir = os.path.join(args.out_dir, args.ckpt_tag)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.pkl_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"[info] ckpt_tag      = {args.ckpt_tag}")
    print(f"[info] checkpoint    = {ckpt_path}")
    print(f"[info] sample source = {sample_source}")
    if eval_only:
        print(f"[info] mode          = eval-only (no pkl / image output)")
    else:
        print(f"[info] output pkl    = {os.path.join(args.pkl_dir, pkl_name)}")

    # Load classifier
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
    classifier.eval()

    # Load datasets
    train_set, dev_set, test_set = load_fashion_mnist("../data", random=random)
    test_loader  = DataLoader(test_set,  batch_size=1000)
    train_loader = DataLoader(train_set, batch_size=1000, shuffle=False)

    test_acc = evaluate(classifier, test_loader, device)
    print(f"[info] test accuracy = {test_acc:.2f}%")

    # --- 1) Obtain correctly classified samples --------------------------
    if sample_source == "test_set":
        x_correct, y_correct = collect_correct_from_loader(
            classifier, test_loader, device, num=args.num_samples)
        print(f"[info] collected {x_correct.shape[0]} correct samples from test set")
    elif sample_source == "train_set":
        x_correct, y_correct = collect_correct_from_loader(
            classifier, train_loader, device, num=args.num_samples)
        print(f"[info] collected {x_correct.shape[0]} correct samples from train set")
    else:
        x_correct, y_correct = load_samples_from_pkl(sample_source)
        print(f"[info] loaded {x_correct.shape[0]} samples from {sample_source}")

    # --- 2) Targeted PGD attack: y -> (y + 1) % 10 ----------------------
    y_target = (y_correct + 1) % 10

    adv, adv_pred = attack_in_batches(
        classifier, x_correct, y_target, device,
        batch_size=200, eps=args.eps, alpha=args.alpha, steps=args.steps,
    )
    success_mask = (adv_pred == y_target)
    success_rate = 100.0 * success_mask.sum().item() / len(success_mask)
    print(f"[result] white-box targeted PGD success rate = "
          f"{success_mask.sum().item()}/{len(success_mask)} = {success_rate:.2f}%")

    # --- cnn_adv 模式：只报告成功率，直接返回 ----------------------------
    if eval_only:
        print(f"[done] test_acc={test_acc:.2f}% | attack_success={success_rate:.2f}%")
        return

    # --- 3) Save ALL successful adversarial samples to pkl ---------------
    pkl_out = os.path.join(args.pkl_dir, pkl_name)
    success_idx_all = success_mask.nonzero(as_tuple=False).flatten()
    save_successful_pkl(
        adv[success_idx_all],
        y_correct[success_idx_all],   # 真实标签，用于对抗训练
        pkl_out,
    )

    # --- 4) Save 10 random successful pairs as images + manifest ---------
    success_idx = success_idx_all.tolist()
    random.shuffle(success_idx)
    pick = success_idx[:10]
    if len(pick) < 10:
        print(f"[warn] only {len(pick)} successful samples, using all")

    sel_orig      = x_correct[pick]
    sel_adv       = adv[pick]
    sel_orig_pred = y_correct[pick]
    sel_adv_pred  = adv_pred[pick]
    sel_target    = y_target[pick]

    grid_path = os.path.join(args.out_dir, "samples_grid.png")
    save_sample_grid(sel_orig, sel_adv, sel_orig_pred, sel_adv_pred, grid_path)
    print(f"[info] saved grid figure to {grid_path}")

    manifest = {
        "ckpt_tag": args.ckpt_tag,
        "checkpoint": os.path.abspath(ckpt_path),
        "sample_source": sample_source,
        "test_accuracy": test_acc,
        "num_attacked": int(x_correct.shape[0]),
        "attack": {
            "type": "targeted PGD (L_inf)",
            "epsilon_pixels": args.eps,
            "alpha_pixels": args.alpha,
            "steps": args.steps,
            "target_mapping": "y -> (y + 1) % 10",
        },
        "success_rate_percent": success_rate,
        "num_successful": int(success_mask.sum().item()),
        "successful_samples_pkl": os.path.abspath(pkl_out),
        "pkl_label_type": "true label (for adversarial training)",
        "samples": [],
    }
    for k, idx in enumerate(pick):
        orig_path = os.path.join(args.out_dir, f"sample_{k:02d}_orig.png")
        adv_path  = os.path.join(args.out_dir, f"sample_{k:02d}_adv.png")
        plt.imsave(orig_path, sel_orig[k].reshape(28, 28).numpy(),
                   cmap="gray", vmin=0, vmax=255)
        plt.imsave(adv_path,  sel_adv[k].reshape(28, 28).numpy(),
                   cmap="gray", vmin=0, vmax=255)
        manifest["samples"].append({
            "orig_image":        os.path.basename(orig_path),
            "adv_image":         os.path.basename(adv_path),
            "orig_label":        int(sel_orig_pred[k].item()),
            "orig_label_name":   CLASS_NAMES[int(sel_orig_pred[k].item())],
            "target_label":      int(sel_target[k].item()),
            "target_label_name": CLASS_NAMES[int(sel_target[k].item())],
            "adv_pred":          int(sel_adv_pred[k].item()),
            "adv_pred_name":     CLASS_NAMES[int(sel_adv_pred[k].item())],
            "linf_perturbation_pixels":
                float((sel_adv[k] - sel_orig[k]).abs().max().item()),
        })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[info] wrote manifest to {manifest_path}")
    print(f"[done] test_acc={test_acc:.2f}% | attack_success={success_rate:.2f}%")


if __name__ == "__main__":
    main()
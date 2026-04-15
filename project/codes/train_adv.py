#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial training on Fashion-MNIST CNN.

在原始训练流程基础上，将白盒攻击生成的对抗样本
（best_whitebox_sample.pkl）混入训练集，提升模型鲁棒性。
保存最佳检查点到 model/cnn_adv.ckpt。
"""
import os
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from fmnist_dataset import load_fashion_mnist
from model import CNN


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def evaluate(classifier, loader, device):
    classifier.eval()
    n, c = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            c += (torch.argmax(logits, dim=1) == y).sum().item()
            n += len(y)
    return 100.0 * c / n


def load_adv_dataset(pkl_path):
    """
    加载白盒攻击生成的对抗样本 pkl。
    格式：[x_array (N,784) float32, y_array (N,) int64]
    返回：TensorDataset，与原始训练集格式一致。
    """
    with open(pkl_path, "rb") as f:
        x_np, y_np = pickle.load(f)
    x = torch.tensor(x_np, dtype=torch.float32)   # (N, 784), [0, 255]
    y = torch.tensor(y_np, dtype=torch.long)       # (N,)
    print(f"[info] loaded {len(x)} adversarial samples from {pkl_path}")
    print(f"[info] adv label distribution: "
          + ", ".join(f"{i}:{(y==i).sum().item()}" for i in range(10)))
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# 主训练逻辑
# ---------------------------------------------------------------------------
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    adv_pkl   = "../attack_data/best_whitebox_train_sample.pkl"
    save_path = "../model/cnn_adv.ckpt"
    os.makedirs("../model", exist_ok=True)

    # ── 1. 加载数据集 ──────────────────────────────────────────────────────
    train_set, dev_set, test_set = load_fashion_mnist("../data", random=random)
    adv_set = load_adv_dataset(adv_pkl)

    # 原始训练集 + 对抗样本合并
    combined_train = ConcatDataset([train_set, adv_set])
    print(f"[info] original train size : {len(train_set)}")
    print(f"[info] adversarial set size: {len(adv_set)}")
    print(f"[info] combined train size : {len(combined_train)}")

    train_loader = DataLoader(combined_train, batch_size=64,
                              shuffle=True, drop_last=True)
    dev_loader   = DataLoader(dev_set,  batch_size=1000)
    test_loader  = DataLoader(test_set, batch_size=1000)

    # ── 2. 初始化模型（从头训练，不依赖旧权重）──────────────────────────────
    classifier = CNN().to(device)
    optimizer  = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion  = nn.CrossEntropyLoss()

    # ── 3. 训练 ───────────────────────────────────────────────────────────
    best_dev = 0.0
    epochs   = 25
    for ep in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_acc = evaluate(classifier, dev_loader, device)
        print(f"Ep {ep:2d}  loss = {total_loss / len(train_loader):.4f}"
              f"  dev acc = {dev_acc:.2f}%")

        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(classifier.state_dict(), save_path)
            print(f"         => saved best model (dev={best_dev:.2f}%)")

    # ── 4. 最终评估 ────────────────────────────────────────────────────────
    classifier.load_state_dict(torch.load(save_path, map_location=device))
    test_acc = evaluate(classifier, test_loader, device)
    print(f"\n[result] best dev acc = {best_dev:.2f}%  |  test acc = {test_acc:.2f}%")
    print(f"[info]   model saved to {save_path}")


if __name__ == "__main__":
    main()
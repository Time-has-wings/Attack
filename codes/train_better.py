#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Fashion-MNIST CNN with enough epochs to exceed 90% test accuracy.
Saves the best checkpoint (by dev acc) to model/cnn_best.ckpt.
"""
import os
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from fmnist_dataset import load_fashion_mnist
from model import CNN


def evaluate(classifier, dataset, device):
    classifier.eval()
    n, c = 0, 0
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            c += (torch.argmax(logits, dim=1) == y).sum().item()
            n += len(y)
    return 100.0 * c / n


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    train_set, dev_set, test_set = load_fashion_mnist("../data", random=random)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=1000)
    test_loader = DataLoader(test_set, batch_size=1000)

    classifier = CNN().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    save_path = "../model/cnn_best.ckpt"
    best_dev = 0.0
    epochs = 15 # 设置为15
    for ep in range(1, epochs + 1):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(x), y)
            loss.backward()
            optimizer.step()
        dev_acc = evaluate(classifier, dev_loader, device)
        print(f"Ep {ep:2d}  dev acc = {dev_acc:.2f}%")
        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(classifier.state_dict(), save_path)

    classifier.load_state_dict(torch.load(save_path))
    test_acc = evaluate(classifier, test_loader, device)
    print(f"Best dev acc = {best_dev:.2f}%, test acc = {test_acc:.2f}%")


if __name__ == "__main__":
    main()

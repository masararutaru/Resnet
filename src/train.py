import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import set_seed, get_device, accuracy
from .datasets import get_dataloaders
from .models import MLP_MNIST, SimpleCNN_CIFAR10, DeepCNN_CIFAR10, ResNet_CIFAR10
import yaml


def build_model(task: str, num_classes: int):
    if task == "mnist-mlp":
        return MLP_MNIST(num_classes)
    elif task == "cifar10-cnn":
        return SimpleCNN_CIFAR10(num_classes)
    elif task == "cifar10-deepcnn":
        return DeepCNN_CIFAR10(num_classes)
    elif task == "cifar10-resnet":
        return ResNet_CIFAR10(num_classes)
    else:
        raise ValueError(task)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc  += accuracy(logits.detach(), y) * bsz
        n += bsz
    return total_loss / n, total_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_acc  += accuracy(logits, y) * bsz
            n += bsz
    return total_loss / n, total_acc / n


def main(cfg_path: str = "configs/default.yaml"):
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    dataset = "mnist" if cfg["task"] == "mnist-mlp" else "cifar10"
    train_loader, test_loader, num_classes = get_dataloaders(dataset, cfg["batch_size"], cfg.get("num_workers", 2))

    model = build_model(cfg["task"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])

    scheduler = None
    if int(cfg.get("step_size", 0)) > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg.get("gamma", 0.1))

    best = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        if scheduler:
            scheduler.step()
        if te_acc > best:
            best = te_acc
            os.makedirs("ckpts", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "acc": best,
                "epoch": epoch,
                "cfg": cfg,
            }, f"ckpts/{cfg['task']}_best.pth")
        print(f"[Epoch {epoch:02d}] train {tr_loss:.4f}/{tr_acc*100:.2f}% | val {te_loss:.4f}/{te_acc*100:.2f}% (best {best*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.cfg) 
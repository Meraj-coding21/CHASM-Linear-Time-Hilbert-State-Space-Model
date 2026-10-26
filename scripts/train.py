#!/usr/bin/env python3
"""Train CHASM with the three-stage optimization protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chasm.constants import CLASS_NAMES
from chasm.data import FolderDataset
from chasm.model import CHASM


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def apply_structured_init(model: nn.Module):
    """Initialize non-backbone CHASM modules."""
    for name, module in model.named_modules():
        if "backbone" in name:
            continue
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels:
                nn.init.xavier_uniform_(module.weight, gain=0.5)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def protect_backbone_batchnorm(backbone: nn.Module):
    """Keep pretrained MobileNetV4 BN statistics and affine parameters frozen."""
    for module in backbone.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)


def hybrid_loss(main, aux1, aux2, targets, criterion, aux_weight=0.1, distill_weight=0.05):
    loss = criterion(main, targets) + aux_weight * (
        criterion(aux1, targets) + criterion(aux2, targets)
    )
    temperature = 2.0
    main_probs = torch.softmax(main.detach().clamp(-50, 50).float() / temperature, dim=1)
    aux2_log = torch.log_softmax(aux2.clamp(-50, 50).float() / temperature, dim=1)
    distill = F.kl_div(aux2_log, main_probs, reduction="batchmean")
    if torch.isfinite(distill):
        loss = loss + distill_weight * distill
    return loss


def make_loaders(root: Path, batch_size: int, workers: int, device: torch.device):
    datasets = {s: FolderDataset(root, s, train_transform=(s == "train")) for s in ("train", "val")}
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=device.type == "cuda"),
    }


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels, _ in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            labels = labels.to(device)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return correct / max(total, 1)


def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    protect_backbone_batchnorm(model.backbone)
    total_loss = total = correct = 0
    for images, labels, _ in tqdm(loader, leave=False, desc="train"):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            main, aux1, aux2 = model(images, return_aux=True)
            loss = hybrid_loss(main, aux1, aux2, labels, criterion)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 0.5)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        correct += int((main.argmax(1) == labels).sum())
        total += len(labels)
    return total_loss / total, correct / total


def run_phase(model, loader_train, loader_val, optimizer, criterion, scaler, device,
              epochs, best, checkpoint, scheduler=None, patience=None, history=None):
    bad = 0
    for _ in range(epochs):
        train_loss, train_acc = train_epoch(model, loader_train, optimizer, scaler, criterion, device)
        val_acc = accuracy(model, loader_val, device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        history.append({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
        if patience is not None and bad >= patience:
            break
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Preprocessed train/val/test root.")
    parser.add_argument("--output", type=Path, default=Path("CHASM_trained.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(args.data_root, args.batch_size, args.workers, device)
    model = CHASM(pretrained_backbone=True).to(device)
    apply_structured_init(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = -1.0

    # Phase 1: backbone frozen, 8 epochs.
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    best = run_phase(model, loaders["train"], loaders["val"], optimizer, criterion, scaler,
                     device, 8, best, args.output, history=history)

    # Phase 2: deepest 20% of backbone parameter tensors unfrozen, 12 epochs.
    backbone = list(model.backbone.named_parameters())
    for _, p in backbone[-int(0.2 * len(backbone)):]:
        p.requires_grad_(True)
    protect_backbone_batchnorm(model.backbone)
    bb_trainable = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
    other = [p for name, p in model.named_parameters() if p.requires_grad and "backbone" not in name]
    optimizer = torch.optim.AdamW([
        {"params": bb_trainable, "lr": 1e-5},
        {"params": other, "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    best = run_phase(model, loaders["train"], loaders["val"], optimizer, criterion, scaler,
                     device, 12, best, args.output, scheduler=scheduler, history=history)

    # Phase 3: full backbone convolutional fine-tuning; backbone BN remains protected.
    for p in model.parameters():
        p.requires_grad_(True)
    protect_backbone_batchnorm(model.backbone)
    bb = list(model.backbone.parameters())
    s1, s2 = len(bb) // 3, 2 * len(bb) // 3
    other = [p for name, p in model.named_parameters() if "backbone" not in name]
    optimizer = torch.optim.AdamW([
        {"params": bb[:s1], "lr": 5e-6},
        {"params": bb[s1:s2], "lr": 1e-5},
        {"params": bb[s2:], "lr": 3e-5},
        {"params": other, "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    best = run_phase(model, loaders["train"], loaders["val"], optimizer, criterion, scaler,
                     device, 15, best, args.output, scheduler=scheduler, patience=10, history=history)

    Path(str(args.output) + ".history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation accuracy: {best:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

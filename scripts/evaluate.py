#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chasm.constants import CLASS_NAMES
from chasm.data import FolderDataset
from chasm.metrics import classification_metrics
from chasm.model import load_chasm_checkpoint, parameter_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Preprocessed split root containing train/val/test folders.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "CHASM.pth")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("evaluation_metrics.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_chasm_checkpoint(args.checkpoint, device=device)
    dataset = FolderDataset(args.data_root, args.split, train_transform=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels, _ in tqdm(loader, desc=f"Evaluating {args.split}"):
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred), len(CLASS_NAMES))
    metrics["num_images"] = len(y_true)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["parameter_count_training_model"] = parameter_count(model, include_auxiliary=True)
    metrics["parameter_count_inference_without_aux_heads"] = parameter_count(model, include_auxiliary=False)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

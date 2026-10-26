#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_SHA256 = "8097f61e4bdf71ef08a9c7263de7ff7c52bcf3b76cd0f5e9a636e2f7d1256b94"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "CHASM.pth")
    parser.add_argument("--full-model", action="store_true",
                        help="Also instantiate timm MobileNetV4 and run a dummy forward pass.")
    args = parser.parse_args()

    digest = sha256(args.checkpoint)
    print("SHA256:", digest)
    if digest != EXPECTED_SHA256:
        raise RuntimeError("Checkpoint SHA256 does not match the released CHASM checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "pos_encoding.pos_embed": (1, 512, 192),
        "shared_mamba.in_proj.weight": (576, 192),
        "main_head.2.weight": (9, 192),
        "hilbert_indexer.indices_dir1": (49,),
    }
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise RuntimeError(f"Unexpected checkpoint tensor for {key}")
    print("Checkpoint verification: PASS")

    if args.full_model:
        from chasm.model import load_chasm_checkpoint
        model = load_chasm_checkpoint(args.checkpoint, device="cpu")
        with torch.inference_mode():
            out = model(torch.zeros(1, 3, 224, 224))
        assert tuple(out.shape) == (1, 9)
        print("Strict state-dict load + dummy forward: PASS")


if __name__ == "__main__":
    main()

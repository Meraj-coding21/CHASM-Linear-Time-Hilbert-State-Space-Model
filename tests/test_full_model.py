from pathlib import Path
import pytest
import torch

pytest.importorskip("timm")
from chasm.model import load_chasm_checkpoint, parameter_count

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_load_and_forward():
    model = load_chasm_checkpoint(ROOT / "weights" / "CHASM.pth", device="cpu")
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert tuple(logits.shape) == (1, 9)
    assert parameter_count(model, include_auxiliary=True) == 3_404_779
    assert parameter_count(model, include_auxiliary=False) == 3_394_393
    assert model.pos_encoding.pos_embed.numel() == 98_304

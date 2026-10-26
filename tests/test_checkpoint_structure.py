from pathlib import Path
import hashlib
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights" / "CHASM.pth"
EXPECTED_SHA = "8097f61e4bdf71ef08a9c7263de7ff7c52bcf3b76cd0f5e9a636e2f7d1256b94"


def test_checkpoint_sha256():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_SHA


def test_checkpoint_critical_shapes():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert tuple(state["pos_encoding.pos_embed"].shape) == (1, 512, 192)
    assert state["pos_encoding.pos_embed"].numel() == 98304
    assert tuple(state["main_head.2.weight"].shape) == (9, 192)
    assert tuple(state["hilbert_indexer.indices_dir1"].shape) == (49,)
    assert torch.equal(state["shared_mamba.norm.weight"], torch.ones(288))
    assert torch.equal(state["shared_mamba.norm.bias"], torch.zeros(288))

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("CHASM requires timm. Install dependencies with: pip install -r requirements.txt") from exc

from .constants import BACKBONE_NAME, D_MODEL, D_STATE, CHUNK_SIZE, NUM_CLASSES
from .hilbert import HilbertIndexer


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, in_channels, 1)
        self.conv_w = nn.Conv2d(hidden, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class AiCGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.coord_attention = CoordinateAttention(in_channels, reduction)
        self.gate_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate_proj(x * self.coord_attention(x))


class PositionalStateEncoding(nn.Module):
    """Trainable 512-position embedding initialized sinusoidally.

    CHASM uses only the first 49 positions for its 7x7 Hilbert sequences.
    The full trainable tensor contains 512*192 = 98,304 parameters.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        with torch.no_grad():
            pos = torch.arange(self.max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, self.d_model, 2).float()
                * -(math.log(10000.0) / self.d_model)
            )
            self.pos_embed[0, :, 0::2] = torch.sin(pos * div)
            self.pos_embed[0, :, 1::2] = torch.cos(pos * div[: self.d_model // 2])
            self.pos_embed.mul_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected BxLx{self.d_model}, got {tuple(x.shape)}")
        length = x.shape[1]
        if length > self.max_len:
            raise ValueError(f"Sequence length {length} exceeds max_len={self.max_len}")
        return x + self.pos_embed[:, :length, :]


def segsum_and_exp(a_chunk: torch.Tensor) -> torch.Tensor:
    """Stable lower-triangular segment-sum transform."""
    cs = torch.cumsum(a_chunk, dim=0)
    diff = (cs.unsqueeze(1) - cs.unsqueeze(0)).clamp(min=-20.0, max=0.0)
    mask = torch.tril(
        torch.ones(diff.shape[0], diff.shape[1], device=diff.device, dtype=torch.bool)
    )
    diff = diff.masked_fill(~mask.unsqueeze(-1), -20.0)
    return torch.exp(diff)


class Mamba2SSDBlock(nn.Module):
    """Shared SSD reasoning block used by CHASM."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 3,
        expand: float = 1.5,
        chunk_size: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.chunk_size = int(chunk_size)
        self.num_heads = int(num_heads)
        self.d_inner = int(expand * d_model)
        if self.d_inner % self.num_heads != 0:
            raise ValueError("Expanded dimension must be divisible by num_heads")
        self.d_head = self.d_inner // self.num_heads

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.B_proj = nn.Linear(self.d_inner, self.num_heads * self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.num_heads * self.d_state, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
        with torch.no_grad():
            self.A_log.uniform_(-4.0, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        batch, original_length, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"Expected feature dimension {self.d_model}, got {dim}")

        pad_len = (self.chunk_size - original_length % self.chunk_size) % self.chunk_size
        padded_length = original_length + pad_len
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        pad_mask = torch.ones(batch, padded_length, 1, device=x.device, dtype=torch.float32)
        if pad_len:
            pad_mask[:, original_length:, :] = 0.0

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :padded_length].transpose(1, 2)
        x_act = F.silu(x_conv.clamp(-50.0, 50.0))
        y = self._ssd_chunk(x_act, pad_mask)
        y = y * F.silu(z.clamp(-50.0, 50.0))
        if pad_len:
            y = y[:, :original_length, :]
        return self.out_proj(y).to(original_dtype)

    def _ssd_chunk(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        chunks = length // self.chunk_size
        x_chunks = x.reshape(batch, chunks, self.chunk_size, self.d_inner)
        x_heads = x_chunks.reshape(
            batch, chunks, self.chunk_size, self.num_heads, self.d_head
        )
        mask_chunks = pad_mask.reshape(batch, chunks, self.chunk_size, 1)

        a = -torch.exp(self.A_log).clamp(min=1e-6)
        l_matrix = segsum_and_exp(a.unsqueeze(0).expand(self.chunk_size, -1))

        x_flat = x_chunks.reshape(batch * chunks, self.chunk_size, self.d_inner)
        b_mat = self.B_proj(x_flat).reshape(
            batch, chunks, self.chunk_size, self.num_heads, self.d_state
        )
        c_mat = self.C_proj(x_flat).reshape(
            batch, chunks, self.chunk_size, self.num_heads, self.d_state
        )
        mask = mask_chunks.unsqueeze(-1)
        b_mat = (b_mat * mask).clamp(-32.0, 32.0)
        c_mat = (c_mat * mask).clamp(-32.0, 32.0)

        bx = torch.einsum("bcthn,bcthd->bcthnd", b_mat, x_heads).clamp(-256.0, 256.0)
        lbx = torch.einsum("ijh,bcjhnd->bcihnd", l_matrix, bx)
        y_heads = torch.einsum("bcihn,bcihnd->bcihd", c_mat, lbx)
        return y_heads.reshape(batch, length, self.d_inner).clamp(-1e4, 1e4)


class SASF(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, maps: List[torch.Tensor]) -> torch.Tensor:
        if len(maps) != 4:
            raise ValueError("SASF expects exactly four reconstructed scan maps")
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = sum(weight * fmap for weight, fmap in zip(weights, maps))
        return self.refine_conv(fused)


class FGA(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_gen = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )

    def forward(self, mamba_feat: torch.Tensor, spatial_shortcut: torch.Tensor) -> torch.Tensor:
        if mamba_feat.shape != spatial_shortcut.shape:
            raise ValueError(
                f"FGA inputs must have identical shapes, got {tuple(mamba_feat.shape)} "
                f"and {tuple(spatial_shortcut.shape)}"
            )
        gate = self.gate_gen(mamba_feat)
        return self.fusion_proj(spatial_shortcut * gate + mamba_feat)


class CHASM(nn.Module):
    """Contiguous Hilbert Aggregated State-space Model (9-class release)."""

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        num_classes: int = NUM_CLASSES,
        d_model: int = D_MODEL,
        d_state: int = D_STATE,
        chunk_size: int = CHUNK_SIZE,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.num_heads = 4
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=[-1],
        )

        # Shape discovery is local and never downloads weights when pretrained_backbone=False.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feature = self.backbone(dummy)[-1]
        self.backbone.train(was_training)
        bb_ch, feat_h, feat_w = feature.shape[1:]
        if (bb_ch, feat_h, feat_w) != (960, 7, 7):
            raise RuntimeError(
                "Unexpected MobileNetV4 feature shape. CHASM expects 960x7x7, "
                f"got {bb_ch}x{feat_h}x{feat_w}. Check the timm/backbone version."
            )

        self.aic_gate = AiCGate(bb_ch)
        self.hilbert_indexer = HilbertIndexer(feat_h, feat_w)
        self.to_mamba_proj = nn.Conv2d(bb_ch, d_model, 1, bias=False)
        self.shared_mamba = Mamba2SSDBlock(
            d_model, d_state=d_state, d_conv=3, expand=1.5,
            chunk_size=chunk_size, num_heads=self.num_heads,
        )
        self.pos_encoding = PositionalStateEncoding(d_model, max_len=512)
        self.sasf = SASF(d_model)
        self.fga = FGA(d_model)
        self.stage1_proj = nn.Conv2d(bb_ch, d_model, 1, bias=False)
        self.aux_head1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bb_ch, num_classes))
        self.aux_head2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d_model, num_classes))
        self.main_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.backbone(x)[-1]
        gated = self.aic_gate(features)
        aux1_logits = self.aux_head1(gated)
        spatial_shortcut = self.stage1_proj(gated)

        mamba_input = self.to_mamba_proj(gated)
        scans = self.hilbert_indexer.scan_4way(mamba_input)
        mamba_outs = [self.shared_mamba(self.pos_encoding(scan)) for scan in scans]
        reconstructed = [
            self.hilbert_indexer.reconstruct_from_scan(out, direction)
            for direction, out in enumerate(mamba_outs)
        ]
        fused_mamba = self.sasf(reconstructed)
        aux2_logits = self.aux_head2(fused_mamba)
        final_feat = self.fga(fused_mamba, spatial_shortcut)
        main_logits = self.main_head(final_feat)
        if return_aux:
            return main_logits, aux1_logits, aux2_logits
        return main_logits


def load_chasm_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> CHASM:
    """Instantiate CHASM without network access and load a checkpoint."""
    device = torch.device(device)
    model = CHASM(pretrained_backbone=False)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}; unexpected={unexpected}")
    model.to(device).eval()
    return model


def parameter_count(model: nn.Module, include_auxiliary: bool = True) -> int:
    if include_auxiliary:
        return sum(p.numel() for p in model.parameters())
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if not (name.startswith("aux_head1.") or name.startswith("aux_head2."))
    )

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from torch import nn


class HilbertCurve:
    """Integer Hilbert traversal on the smallest enclosing power-of-two grid."""

    @staticmethod
    def _rotate(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    @staticmethod
    def d2xy(n: int, d: int) -> Tuple[int, int]:
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            x, y = HilbertCurve._rotate(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    @staticmethod
    def generate_traversal(height: int, width: int) -> np.ndarray:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        n = 2 ** int(np.ceil(np.log2(max(height, width))))
        points = []
        for d in range(n * n):
            x, y = HilbertCurve.d2xy(n, d)
            if x < width and y < height:
                points.append((y, x))
        traversal = np.asarray(points, dtype=np.int64)
        if traversal.shape != (height * width, 2):
            raise RuntimeError("Hilbert traversal did not cover every spatial position exactly once")
        return traversal


class HilbertIndexer(nn.Module):
    """Four fixed 7x7-compatible Hilbert permutations used by CHASM.

    The index buffers have no learnable parameters. For CHASM's square 7x7
    feature map, the transforms correspond to base, 90-degree orientation,
    180-degree orientation, and horizontal mirror.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.H = int(height)
        self.W = int(width)
        self.L = self.H * self.W
        base = HilbertCurve.generate_traversal(self.H, self.W)
        r, c = base[:, 0], base[:, 1]

        if self.H != self.W:
            raise ValueError(
                "The released CHASM checkpoint uses a square 7x7 feature map; "
                "non-square inputs are intentionally rejected to avoid ambiguous rotations."
            )

        directions = {
            "dir1": (r, c),
            "dir2": (c, self.H - 1 - r),
            "dir3": (self.H - 1 - r, self.W - 1 - c),
            "dir4": (r, self.W - 1 - c),
        }
        for name, (rows, cols) in directions.items():
            idx = torch.from_numpy((rows * self.W + cols).astype(np.int64))
            self._validate_permutation(idx, name)
            self.register_buffer(f"indices_{name}", idx)
            self.register_buffer(f"reverse_{name}", self._reverse(idx))

    def _validate_permutation(self, idx: torch.Tensor, name: str) -> None:
        expected = torch.arange(self.L, dtype=idx.dtype)
        if idx.numel() != self.L or not torch.equal(torch.sort(idx).values.cpu(), expected):
            raise RuntimeError(f"{name} is not a valid permutation of 0..{self.L - 1}")

    @staticmethod
    def _reverse(fwd: torch.Tensor) -> torch.Tensor:
        rev = torch.zeros_like(fwd)
        rev[fwd] = torch.arange(len(fwd), dtype=fwd.dtype, device=fwd.device)
        return rev

    def scan_4way(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if (height, width) != (self.H, self.W):
            raise ValueError(f"Expected spatial size {self.H}x{self.W}, got {height}x{width}")
        flat = x.reshape(batch, channels, -1)
        scans = []
        for name in ("dir1", "dir2", "dir3", "dir4"):
            idx = getattr(self, f"indices_{name}")
            expanded = idx.view(1, 1, -1).expand(batch, channels, -1)
            scans.append(torch.gather(flat, 2, expanded).transpose(1, 2))
        return scans

    def _validate_scan(self, x_scan: torch.Tensor, direction: int):
        if direction not in (0, 1, 2, 3):
            raise ValueError("direction must be one of 0, 1, 2, 3")
        if x_scan.ndim != 3 or x_scan.shape[1] != self.L:
            raise ValueError(f"Expected Bx{self.L}xC scan, got {tuple(x_scan.shape)}")

    def reconstruct_from_scan(self, x_scan: torch.Tensor, direction: int) -> torch.Tensor:
        """Reconstruct a CHASM scan into its 2D spatial feature map."""
        self._validate_scan(x_scan, direction)
        batch, _, channels = x_scan.shape
        flat = x_scan.transpose(1, 2)
        rev = [self.reverse_dir1, self.reverse_dir2, self.reverse_dir3, self.reverse_dir4][direction]
        expanded = rev.view(1, 1, -1).expand(batch, channels, -1)
        out = torch.zeros_like(flat)
        out.scatter_(2, expanded, flat)
        return out.reshape(batch, channels, self.H, self.W)

    def inverse_scan(self, x_scan: torch.Tensor, direction: int) -> torch.Tensor:
        """Return the exact inverse permutation of :meth:`scan_4way`."""
        self._validate_scan(x_scan, direction)
        batch, _, channels = x_scan.shape
        flat = x_scan.transpose(1, 2)
        fwd = [self.indices_dir1, self.indices_dir2, self.indices_dir3, self.indices_dir4][direction]
        expanded = fwd.view(1, 1, -1).expand(batch, channels, -1)
        out = torch.zeros_like(flat)
        out.scatter_(2, expanded, flat)
        return out.reshape(batch, channels, self.H, self.W)

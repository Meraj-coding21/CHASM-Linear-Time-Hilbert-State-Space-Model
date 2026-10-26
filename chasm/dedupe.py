from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.fft import dctn


def phash64(path: str | Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Compute a 64-bit perceptual hash without an external imagehash dependency."""
    path = Path(path)
    side = hash_size * highfreq_factor
    with Image.open(path) as image:
        image = image.convert("L").resize((side, side), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
    coeff = dctn(pixels, type=2, norm="ortho")[:hash_size, :hash_size]
    flat = coeff.flatten()
    median = np.median(flat[1:])  # exclude DC term when determining threshold
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def hamming(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class BKTree:
    """BK-tree for Hamming-distance search over integer hashes."""

    def __init__(self):
        self.root = None

    def add(self, value: int, index: int) -> None:
        node = [value, [(index)], {}]
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            distance = hamming(value, cur[0])
            if distance == 0:
                cur[1].append(index)
                return
            child = cur[2].get(distance)
            if child is None:
                cur[2][distance] = node
                return
            cur = child

    def query(self, value: int, max_distance: int) -> List[int]:
        if self.root is None:
            return []
        found = []
        stack = [self.root]
        while stack:
            cur = stack.pop()
            distance = hamming(value, cur[0])
            if distance <= max_distance:
                found.extend(cur[1])
            lower, upper = distance - max_distance, distance + max_distance
            for edge, child in cur[2].items():
                if lower <= edge <= upper:
                    stack.append(child)
        return found


@dataclass(frozen=True)
class DuplicateCluster:
    cluster_id: int
    indices: Tuple[int, ...]


def cluster_hashes(hashes: Sequence[int], threshold: int = 4) -> List[DuplicateCluster]:
    """Connected components under pHash Hamming distance <= threshold."""
    uf = UnionFind(len(hashes))
    tree = BKTree()
    for index, value in enumerate(hashes):
        for other in tree.query(value, threshold):
            uf.union(index, other)
        tree.add(value, index)

    grouped: Dict[int, List[int]] = {}
    for index in range(len(hashes)):
        grouped.setdefault(uf.find(index), []).append(index)
    clusters = [
        DuplicateCluster(cluster_id=i, indices=tuple(sorted(indices)))
        for i, indices in enumerate(sorted(grouped.values(), key=lambda xs: xs[0]))
    ]
    return clusters

from pathlib import Path
import numpy as np
from PIL import Image

from chasm.dedupe import phash64, hamming, cluster_hashes


def test_exact_duplicate_hashes_cluster(tmp_path: Path):
    a = (np.arange(64 * 64).reshape(64, 64) % 255).astype(np.uint8)
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.fromarray(a).save(p1)
    Image.fromarray(a).save(p2)
    hashes = [phash64(p1), phash64(p2)]
    assert hamming(*hashes) == 0
    clusters = cluster_hashes(hashes, threshold=4)
    assert len(clusters) == 1
    assert clusters[0].indices == (0, 1)

import torch
from chasm.hilbert import HilbertIndexer


def test_hilbert_permutations_are_complete():
    indexer = HilbertIndexer(7, 7)
    expected = torch.arange(49)
    for direction in range(1, 5):
        idx = getattr(indexer, f"indices_dir{direction}")
        assert torch.equal(torch.sort(idx).values.cpu(), expected)


def test_scan_shapes_and_inverse_permutation():
    indexer = HilbertIndexer(7, 7)
    x = torch.arange(2 * 3 * 7 * 7).reshape(2, 3, 7, 7).float()
    scans = indexer.scan_4way(x)
    assert all(tuple(scan.shape) == (2, 49, 3) for scan in scans)
    for direction, scan in enumerate(scans):
        restored = indexer.inverse_scan(scan, direction)
        assert torch.equal(restored, x)


def test_chasm_reconstruction_shape():
    indexer = HilbertIndexer(7, 7)
    x = torch.arange(49).reshape(1, 1, 7, 7).float()
    scan = indexer.scan_4way(x)[0]
    restored = indexer.reconstruct_from_scan(scan, 0)
    assert tuple(restored.shape) == tuple(x.shape)

#!/usr/bin/env python3
"""Create a leakage-safe train/val/test split using perceptual-hash clusters.

Near-duplicate images are detected on the *raw* images before any preprocessing.
All members of a duplicate cluster are assigned to the same split, so perceptual
near-duplicates cannot cross train/validation/test boundaries.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import shutil
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chasm.constants import CLASS_NAMES, IMAGE_EXTENSIONS
from chasm.dedupe import cluster_hashes, phash64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Raw UIdataGB root containing the nine class folders.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train", type=float, default=0.75)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--phash-threshold", type=int, default=4,
                        help="Maximum 64-bit pHash Hamming distance for near-duplicate clustering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize", choices=["copy", "hardlink", "none"], default="hardlink",
                        help="How to create split folders. hardlink falls back to copy when unavailable.")
    return parser.parse_args()


def collect_images(root: Path):
    present = {p.name for p in root.iterdir() if p.is_dir()}
    missing = [c for c in CLASS_NAMES if c not in present]
    if missing:
        raise RuntimeError("Missing expected UIdataGB class folders: " + ", ".join(missing))
    extras = sorted(present - set(CLASS_NAMES))
    if extras:
        print("Warning: ignoring non-CHASM folders:", extras)
    rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for path in sorted((root / class_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": path.resolve(), "class_name": class_name, "label": class_index})
    if not rows:
        raise RuntimeError(f"No supported images found under {root}")
    return rows


def assign_clusters(rows, clusters, ratios, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for cluster in clusters:
        labels = {rows[i]["class_name"] for i in cluster.indices}
        if len(labels) != 1:
            members = [str(rows[i]["path"]) for i in cluster.indices]
            raise RuntimeError(
                "A perceptual duplicate cluster contains conflicting class labels. "
                "Please inspect these files before splitting:\n" + "\n".join(members)
            )
        by_class[next(iter(labels))].append(cluster)

    assignments = {}
    split_names = ["train", "val", "test"]
    for class_name in CLASS_NAMES:
        class_clusters = by_class[class_name]
        rng.shuffle(class_clusters)
        class_clusters.sort(key=lambda c: len(c.indices), reverse=True)
        total = sum(len(c.indices) for c in class_clusters)
        targets = {s: ratios[j] * total for j, s in enumerate(split_names)}
        counts = {s: 0 for s in split_names}
        for cluster in class_clusters:
            size = len(cluster.indices)
            # Assign to split with largest normalized remaining deficit.
            candidates = []
            for s in split_names:
                target = max(targets[s], 1e-9)
                deficit = (targets[s] - counts[s]) / target
                candidates.append((deficit, -counts[s], s))
            _, _, chosen = max(candidates)
            assignments[cluster.cluster_id] = chosen
            counts[chosen] += size
        print(class_name, {s: counts[s] for s in split_names})
    return assignments


def materialize(src: Path, dst: Path, mode: str):
    if mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-9 or min(ratios) <= 0:
        raise ValueError("train/val/test ratios must be positive and sum to 1")
    if not 0 <= args.phash_threshold <= 64:
        raise ValueError("phash-threshold must be between 0 and 64")

    rows = collect_images(args.data_root)
    print(f"Computing pHash for {len(rows)} images...")
    hashes = [phash64(row["path"]) for row in rows]
    clusters = cluster_hashes(hashes, threshold=args.phash_threshold)
    duplicate_clusters = [c for c in clusters if len(c.indices) > 1]
    print(f"Clusters: {len(clusters)}; near-duplicate clusters: {len(duplicate_clusters)}")

    assignments = assign_clusters(rows, clusters, ratios, args.seed)
    cluster_of = {}
    for cluster in clusters:
        for index in cluster.indices:
            cluster_of[index] = cluster.cluster_id

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "class_name", "label", "source_path", "relative_path",
            "phash_hex", "duplicate_cluster_id", "cluster_size",
        ])
        writer.writeheader()
        cluster_sizes = {c.cluster_id: len(c.indices) for c in clusters}
        for index, row in enumerate(rows):
            cid = cluster_of[index]
            split = assignments[cid]
            # Prefix with stable source index to prevent same-name collisions.
            filename = f"{index:05d}_{row['path'].name}"
            relative = Path(split) / row["class_name"] / filename
            writer.writerow({
                "split": split,
                "class_name": row["class_name"],
                "label": row["label"],
                "source_path": str(row["path"]),
                "relative_path": str(relative),
                "phash_hex": f"{hashes[index]:016x}",
                "duplicate_cluster_id": cid,
                "cluster_size": cluster_sizes[cid],
            })
            materialize(row["path"], args.output_root / relative, args.materialize)

    # Prove that no duplicate cluster crosses splits.
    seen = defaultdict(set)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seen[row["duplicate_cluster_id"]].add(row["split"])
    crossing = {k: v for k, v in seen.items() if len(v) > 1}
    if crossing:
        raise AssertionError(f"Leakage detected: {crossing}")

    counts = Counter(assignments[cluster_of[i]] for i in range(len(rows)))
    print("Final split counts:", dict(counts))
    print("No pHash duplicate cluster crosses train/val/test.")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

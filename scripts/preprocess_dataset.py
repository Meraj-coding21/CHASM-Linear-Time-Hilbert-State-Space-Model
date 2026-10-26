#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chasm.constants import CLASS_NAMES, IMAGE_EXTENSIONS
from chasm.preprocessing import HybridGallbladderDenoiser, load_grayscale, save_grayscale


def main():
    parser = argparse.ArgumentParser(description="Apply CHASM SRAD + wavelet preprocessing.")
    parser.add_argument("--input-root", required=True, type=Path,
                        help="Leakage-safe root produced by make_leakage_safe_split.py")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    denoiser = HybridGallbladderDenoiser(srad_iterations=15, wavelet="db4", dwt_levels=2)
    jobs = []
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            source_dir = args.input_root / split / class_name
            if not source_dir.is_dir():
                raise FileNotFoundError(source_dir)
            for path in sorted(source_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    jobs.append((split, class_name, path))

    for split, class_name, source in tqdm(jobs, desc="SRAD-DTCWT preprocessing"):
        # PNG output avoids introducing additional JPEG artifacts.
        target = args.output_root / split / class_name / f"{source.stem}.png"
        if target.exists() and not args.overwrite:
            continue
        image = load_grayscale(source)
        _, denoised = denoiser.denoise(image)
        save_grayscale(denoised, target)
    print(f"Processed {len(jobs)} images into {args.output_root}")


if __name__ == "__main__":
    main()

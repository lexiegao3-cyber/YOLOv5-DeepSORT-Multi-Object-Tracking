#!/usr/bin/env python3
"""Validate and index a local MOT17/MOT20 dataset.

MOTChallenge data is distributed by the benchmark site.  This utility does
not redistribute the dataset; it validates the standard directory layout and
writes a manifest that can be used to select sequences for tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_split_root(root, dataset, split):
    root = Path(root).expanduser().resolve()
    candidates = [root / dataset / split, root / split]
    if root.name.lower() == split.lower():
        candidates.insert(0, root)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find {dataset}/{split}. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def validate_sequence(sequence_dir, split):
    image_dir = sequence_dir / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    frame_numbers = []
    for image in images:
        try:
            frame_numbers.append(int(image.stem))
        except ValueError as exc:
            raise ValueError(f"MOT image names must be numeric: {image}") from exc
    frame_numbers.sort()
    missing_frames = [
        frame for frame in range(frame_numbers[0], frame_numbers[-1] + 1)
        if frame not in set(frame_numbers)
    ]
    gt_path = sequence_dir / "gt" / "gt.txt"
    if split == "train" and not gt_path.is_file():
        raise FileNotFoundError(f"Missing training annotations: {gt_path}")

    annotation_rows = 0
    if gt_path.is_file():
        with gt_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                fields = line.strip().split(",")
                if len(fields) < 6:
                    raise ValueError(f"Invalid annotation at {gt_path}:{line_number}")
                annotation_rows += 1

    return {
        "name": sequence_dir.name,
        "path": str(sequence_dir),
        "image_dir": str(image_dir),
        "frames": len(images),
        "first_frame": frame_numbers[0],
        "last_frame": frame_numbers[-1],
        "missing_frames": missing_frames,
        "ground_truth": str(gt_path) if gt_path.is_file() else None,
        "annotation_rows": annotation_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MOT data root")
    parser.add_argument("--dataset", choices=("MOT17", "MOT20"), required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sequence", action="append", help="Sequence name; repeat for multiple sequences")
    parser.add_argument("--output", help="Manifest path; defaults to <root>/<dataset>_<split>_manifest.json")
    args = parser.parse_args()

    split_root = resolve_split_root(args.root, args.dataset, args.split)
    available = sorted(path for path in split_root.iterdir() if path.is_dir())
    selected = set(args.sequence or [path.name for path in available])
    unknown = selected - {path.name for path in available}
    if unknown:
        raise FileNotFoundError(
            f"Unknown sequence(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(path.name for path in available)}"
        )

    sequences = [
        validate_sequence(split_root / name, args.split)
        for name in sorted(selected)
    ]
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "root": str(split_root),
        "sequences": sequences,
    }
    output = Path(args.output).expanduser() if args.output else split_root.parent / f"{args.dataset}_{args.split}_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated {len(sequences)} sequence(s); manifest written to {output}")
    for sequence in sequences:
        warning = f" WARNING: {len(sequence['missing_frames'])} missing frames." if sequence["missing_frames"] else ""
        print(f"  {sequence['name']}: {sequence['frames']} frames, {sequence['annotation_rows']} annotation rows.{warning}")


if __name__ == "__main__":
    main()

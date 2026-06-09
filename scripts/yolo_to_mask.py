"""Convert a YOLO segmentation annotation to a binary mask PNG.

Usage:
    python scripts/yolo_to_mask.py \
        --image assets/pencilbag_ref.jpg \
        --label pencilbag_yolo.txt \
        --class-id 0 \
        --output assets/pencilbag_ref_mask.png

YOLO seg format (one line per object):
    class_id x1 y1 x2 y2 x3 y3 ...   (coords normalized 0‑1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--label", required=True, help="YOLO .txt label path")
    parser.add_argument("--class-id", type=int, default=0, help="Class id to extract")
    parser.add_argument("--output", default=None, help="Output mask path")
    args = parser.parse_args()

    # Load image to get dimensions
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    H, W = img.shape[:2]

    # Parse YOLO labels
    mask = np.zeros((H, W), dtype=np.uint8)

    with open(args.label) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id != args.class_id:
                continue

            # Normalized polygon → pixel coordinates
            coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
            xy = coords.reshape(-1, 2) * [W, H]
            xy = xy.round().astype(np.int32)
            cv2.fillPoly(mask, [xy], 255)

    # Save
    output = Path(args.output or args.image.rsplit(".", 1)[0] + "_mask.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    n_objs = len(np.unique(mask)) - 1
    print(f"Mask saved: {output}  ({n_objs} object(s) from class {args.class_id})")


if __name__ == "__main__":
    main()

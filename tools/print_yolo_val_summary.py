#!/usr/bin/env python
"""Run YOLO validation and print the same kind of summary seen at the end of training.

Edit the CONFIG block below, then run this file in the environment that has
`torch` and `ultralytics`.
"""

from __future__ import annotations

import sys
from pathlib import Path


# =========================
# CONFIG: edit this block
# =========================
WEIGHT = r"C:\Users\18377\Desktop\ALL\weights\best.pt"
DATA_YAML = r"D:\yolo26\ultralytics-main\experiments\datasets\visdrone2019_local.yaml"
SPLIT = "val"       # match the original training-time validation split
DEVICE = "0"        # "0" or "cpu"
BATCH = 4
IMGSZ = 640
CONF = 0.001
IOU = 0.7
MAX_DET = 300
PLOTS = True


def ensure_local_ultralytics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    ensure_local_ultralytics()

    from ultralytics import YOLO

    weight_path = Path(WEIGHT).resolve()
    data_yaml = Path(DATA_YAML).resolve()

    if not weight_path.exists():
        raise FileNotFoundError(f"Weight not found: {weight_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    model = YOLO(str(weight_path))

    print(f"[INFO] weight: {weight_path}")
    print(f"[INFO] data:   {data_yaml}")
    print(f"[INFO] split:  {SPLIT}")
    print()

    # Ultralytics will print the validation summary in the terminal,
    # including the per-class table, just like the end of training.
    results = model.val(
        data=str(data_yaml),
        split=SPLIT,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        plots=PLOTS,
        save_json=False,
        verbose=True,
    )

    print()
    print("[INFO] overall metrics")
    for key, value in results.results_dict.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

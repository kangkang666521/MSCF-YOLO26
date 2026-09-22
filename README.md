# MSCF-YOLO26

MSCF-YOLO26 is a YOLO26s-based detector for tiny objects in UAV aerial images. It implements multi-scale shallow feature enhancement, context enhancement, scale-sequence feature fusion, and a P2 high-resolution detection head.

## Architecture

- **MSFE**: multi-receptive-field enhancement at backbone P2, P3, and P4.
- **CEM**: context aggregation and channel recalibration at backbone P5 and after neck fusion.
- **SSFF**: pairwise, content-adaptive fusion along `P5 -> P4 -> P3 -> P2 -> P3 -> P4 -> P5`.
- **P2 head**: four-scale predictions at P2/4, P3/8, P4/16, and P5/32.

The full model configuration is [`ultralytics/cfg/models/26/yolo26-thesis-v1.yaml`](ultralytics/cfg/models/26/yolo26-thesis-v1.yaml). Training settings and a local launcher are documented in [`experiments/README.md`](experiments/README.md).

## Quick start

```powershell
E:\Anaconda\envs\yolo\python.exe tools\run_yolo_local.py train --model ultralytics\cfg\models\26\yolo26-thesis-v1.yaml --weights weights\yolo26s.pt --data experiments\datasets\visdrone2019_local.yaml --epochs 200 --batch 4 --imgsz 640 --workers 2 --name mscf_yolo26_seed0 optimizer=AdamW lr0=0.003 lrf=0.12 momentum=0.937 weight_decay=0.0005 warmup_epochs=5 amp=True seed=0
```

Set the `path` in `experiments/datasets/visdrone2019_local.yaml` to your VisDrone2019 dataset before training.

## Upstream material

This project is based on Ultralytics YOLO26. The retained upstream README, contribution guide, and AGPL-3.0 license are available in [`docs/`](docs/).

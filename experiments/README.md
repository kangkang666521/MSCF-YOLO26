# MSCF-YOLO26 VisDrone experiments

The model and commands in this directory reproduce the architecture and training settings reported for MSCF-YOLO26:

- MSFE is placed at backbone P2, P3 and P4; CEM is placed at backbone P5.
- The neck follows `P5 -> P4 -> P3 -> P2 -> P3 -> P4 -> P5`, using six pairwise SSFF blocks followed by CEM refinement.
- The detector predicts at P2/4, P3/8, P4/16 and P5/32.
- Training uses 640-pixel inputs, 200 epochs, batch size 4, AdamW, `lr0=0.003`, `lrf=0.12`, momentum `0.937`, weight decay `0.0005`, and five warm-up epochs.

Set the `path` entry in `experiments/datasets/visdrone2019_local.yaml` to the root of the locally converted VisDrone2019 dataset before training. The file already defines the paper's ten categories.

Run the following command from the repository root with the local `yolo` environment. The supplied launcher supports regular Ultralytics `key=value` overrides.

```powershell
E:\Anaconda\envs\yolo\python.exe tools\run_yolo_local.py train --model ultralytics\cfg\models\26\yolo26-thesis-v1.yaml --weights weights\yolo26s.pt --data experiments\datasets\visdrone2019_local.yaml --epochs 200 --batch 4 --imgsz 640 --workers 2 --name mscf_yolo26_seed0 optimizer=AdamW lr0=0.003 lrf=0.12 momentum=0.937 weight_decay=0.0005 warmup_epochs=5 amp=True seed=0 mosaic=1.0 hsv_h=0.013 hsv_s=0.65 hsv_v=0.35 translate=0.05 scale=0.4 fliplr=0.5 close_mosaic=15
```

Repeat the full-model run with `seed=1` and `seed=2` to obtain the manuscript's mean and standard deviation. The corresponding ablation models are:

| Experiment | Model YAML |
| --- | --- |
| MSFE | `ultralytics/cfg/models/26/yolo26-msfe-v1.yaml` |
| CEM | `ultralytics/cfg/models/26/yolo26-cem-v1.yaml` |
| MSFE + CEM | `ultralytics/cfg/models/26/yolo26-msfe-cem-v1.yaml` |
| SSFF + P2 | `ultralytics/cfg/models/26/yolo26-ssff-p2-v1.yaml` |
| MSCF-YOLO26 | `ultralytics/cfg/models/26/yolo26-thesis-v1.yaml` |

Validate a trained checkpoint with:

```powershell
E:\Anaconda\envs\yolo\python.exe tools\run_yolo_local.py val --model runs\detect\mscf_yolo26_seed0\weights\best.pt --data experiments\datasets\visdrone2019_local.yaml --batch 4 --imgsz 640 --workers 2 --name mscf_yolo26_seed0_val
```

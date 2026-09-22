"""Run local Ultralytics YOLO training or validation with stable paper-style arguments."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch

from ultralytics import YOLO


def _value(value: str):
    """Interpret ordinary CLI values and key=value overrides without using eval."""
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _backbone_layers(model: YOLO):
    """Return the sequential backbone, ending immediately before the first neck upsample."""
    layers = list(model.model.model)
    return layers[: next(i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Upsample))]


def _transfer_backbone_by_order(model: YOLO, weights: str) -> int:
    """Transfer compatible YOLO26s backbone tensors despite inserted MSCF modules.

    MSCF-YOLO26 inserts MSFE and CEM blocks, so matching state-dictionary keys by numerical layer index initializes only
    a small subset of the backbone. This routine instead walks the source and target backbones in order, skips inserted
    module types, and copies only tensors whose names and shapes match exactly.
    """
    source_layers = _backbone_layers(YOLO(weights))
    target_layers = _backbone_layers(model)
    target_index = 0
    transferred = 0

    for source in source_layers:
        while target_index < len(target_layers) and type(target_layers[target_index]) is not type(source):
            target_index += 1
        if target_index == len(target_layers):
            break

        target = target_layers[target_index]
        source_state = source.state_dict()
        target_state = target.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        target.load_state_dict(compatible, strict=False)
        transferred += len(compatible)
        target_index += 1

    return transferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "val", "predict", "export"))
    parser.add_argument("--model", required=True, help="Model YAML or checkpoint path.")
    parser.add_argument("--weights", help="Optional pretrained checkpoint to partially load into a YAML model.")
    parser.add_argument("--data", help="Dataset YAML path (required for train and val).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="mscf_yolo26")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--device", default=None)
    parser.add_argument("--source", help="Image, video or directory for predict mode.")
    parser.add_argument("overrides", nargs="*", help="Additional Ultralytics overrides as key=value.")
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.weights:
        copied = _transfer_backbone_by_order(model, str(Path(args.weights)))
        print(f"Transferred {copied} compatible backbone tensors from {args.weights}.")

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"Override must use key=value, got: {item}")
        key, value = item.split("=", 1)
        overrides[key] = _value(value)

    common = {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        common["device"] = args.device
    common.update(overrides)

    if args.mode == "train":
        if not args.data:
            parser.error("--data is required for train mode")
        model.train(data=args.data, epochs=args.epochs, **common)
    elif args.mode == "val":
        if not args.data:
            parser.error("--data is required for val mode")
        model.val(data=args.data, **common)
    elif args.mode == "predict":
        if not args.source:
            parser.error("--source is required for predict mode")
        model.predict(source=args.source, **common)
    else:
        model.export(**common)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="为 YOLO 检测模型生成 Grad-CAM 热力图。")
    parser.add_argument("--weights", type=str, required=True, help="训练好的权重路径，例如 best.pt")
    parser.add_argument("--source", type=str, required=True, help="输入图片路径，或包含图片的文件夹路径")
    parser.add_argument("--output", type=str, default="runs/gradcam", help="热力图保存目录")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备，例如 0 或 cpu")
    parser.add_argument("--layer", type=int, default=31, help="挂钩层索引，完整论文模型通常用 31")
    parser.add_argument("--topk", type=int, default=10, help="用前 topk 个最高置信度目标共同反传")
    parser.add_argument("--class-id", type=int, default=None, help="可选。只看某一类的热力图")
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度")
    parser.add_argument("--max-images", type=int, default=10, help="当 source 是文件夹时，最多处理多少张")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device}")


def collect_images(source: str, max_images: int) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        images = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        return images[:max_images]
    raise FileNotFoundError(f"未找到输入路径: {source}")


def preprocess(image_path: Path, imgsz: int, device: torch.device):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True, stride=32)
    resized_bgr = letterbox(image=image_bgr)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return image_bgr, resized_bgr, tensor.to(device)


class ActivationsAndGradients:
    def __init__(self, layer: torch.nn.Module):
        self.activations = None
        self.gradients = None
        self.hooks = [
            layer.register_forward_hook(self.save_activation),
            layer.register_full_backward_hook(self.save_gradient),
        ]

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def build_target(scores: torch.Tensor, topk: int, class_id: int | None):
    if class_id is None:
        per_anchor_scores, class_indices = scores.max(dim=1)
    else:
        per_anchor_scores = scores[:, class_id, :]
        class_indices = torch.full_like(per_anchor_scores, class_id, dtype=torch.long)

    topk = min(topk, per_anchor_scores.shape[-1])
    values, indices = torch.topk(per_anchor_scores[0], k=topk)
    target = values.sum()
    selected_classes = class_indices[0, indices].detach().cpu().tolist()
    return target, selected_classes, values.detach().cpu().tolist()


def make_gradcam(activations: torch.Tensor, gradients: torch.Tensor, out_shape: tuple[int, int]) -> np.ndarray:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=out_shape, mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam


def overlay_heatmap(base_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heat = np.uint8(cam * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)


def ensure_config_dir(project_root: Path):
    config_dir = project_root / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    ensure_config_dir(project_root)

    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    net = model.model.to(device)
    net.eval()
    net.requires_grad_(True)

    if args.layer < 0 or args.layer >= len(net.model):
        raise IndexError(f"layer 超出范围，当前模型共有 {len(net.model)} 层。")

    hook = ActivationsAndGradients(net.model[args.layer])
    image_paths = collect_images(args.source, args.max_images)

    for image_path in image_paths:
        _, resized_bgr, tensor = preprocess(image_path, args.imgsz, device)

        net.zero_grad(set_to_none=True)
        detect_head = net.model[-1]
        detect_training = detect_head.training
        detect_head.train()
        outputs = net(tensor)
        detect_head.train(detect_training)
        scores = outputs["one2many"]["scores"]
        target, selected_classes, selected_scores = build_target(scores, args.topk, args.class_id)
        target.backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("未成功捕获特征图或梯度，请尝试更换 --layer。")

        cam = make_gradcam(hook.activations, hook.gradients, resized_bgr.shape[:2])
        overlay = overlay_heatmap(resized_bgr, cam, args.alpha)

        stem = image_path.stem
        heatmap_path = output_dir / f"{stem}_gradcam.jpg"
        rawcam_path = output_dir / f"{stem}_cam.jpg"

        cv2.imwrite(str(heatmap_path), overlay)
        cv2.imwrite(str(rawcam_path), np.uint8(cam * 255))

        classes_text = ",".join(map(str, selected_classes[:5]))
        scores_text = ",".join(f"{s:.3f}" for s in selected_scores[:5])
        print(f"[OK] {image_path.name}")
        print(f"     save: {heatmap_path}")
        print(f"     top classes: {classes_text}")
        print(f"     top scores : {scores_text}")

    hook.remove()


if __name__ == "__main__":
    main()

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
    parser = argparse.ArgumentParser(description="为 YOLO 检测模型生成按单个检测框反传的 Grad-CAM 热力图。")
    parser.add_argument("--weights", type=str, required=True, help="训练好的权重路径，例如 best.pt")
    parser.add_argument("--source", type=str, required=True, help="输入图片路径，或包含图片的文件夹路径")
    parser.add_argument("--output", type=str, default="runs/gradcam_box", help="热力图保存目录")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备，例如 0 或 cpu")
    parser.add_argument("--layer", type=int, default=22, help="挂钩层索引。微小目标通常先看 22 或 25")
    parser.add_argument("--det-index", type=int, default=0, help="选择第几个检测框，默认 0 表示最高置信度")
    parser.add_argument("--conf", type=float, default=0.25, help="预测时的置信度阈值")
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度")
    parser.add_argument("--max-images", type=int, default=10, help="当 source 是文件夹时，最多处理多少张")
    parser.add_argument("--draw-box", action="store_true", help="是否在热力图上绘制检测框，默认不绘制")
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

    h0, w0 = image_bgr.shape[:2]
    gain = min(imgsz / h0, imgsz / w0)
    new_w, new_h = round(w0 * gain), round(h0 * gain)
    pad_w = (imgsz - new_w) / 2
    pad_h = (imgsz - new_h) / 2
    return resized_bgr, tensor.to(device), gain, pad_w, pad_h


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


def ensure_config_dir(project_root: Path):
    config_dir = project_root / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def box_iou(boxes1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], box2[0])
    y1 = torch.maximum(boxes1[:, 1], box2[1])
    x2 = torch.minimum(boxes1[:, 2], box2[2])
    y2 = torch.minimum(boxes1[:, 3], box2[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (box2[2] - box2[0]).clamp(min=0) * (box2[3] - box2[1]).clamp(min=0)
    return inter / (area1 + area2 - inter + 1e-7)


def convert_box_to_letterbox(
    box_xyxy: np.ndarray, gain: float, pad_w: float, pad_h: float, device: torch.device
) -> torch.Tensor:
    box = torch.tensor(box_xyxy, dtype=torch.float32, device=device)
    box[[0, 2]] = box[[0, 2]] * gain + pad_w
    box[[1, 3]] = box[[1, 3]] * gain + pad_h
    return box


def draw_box(image: np.ndarray, box_xyxy: np.ndarray, text: str):
    x1, y1, x2, y2 = [round(v) for v in box_xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


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


def select_detection(det_model: YOLO, image_path: Path, imgsz: int, conf: float, device: str, det_index: int):
    result = det_model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    order = np.argsort(-confs)
    det_index = min(det_index, len(order) - 1)
    idx = order[det_index]
    return {
        "box": boxes[idx],
        "conf": float(confs[idx]),
        "cls": int(classes[idx]),
        "det_index": det_index,
    }


def main():
    args = parse_args()
    ensure_config_dir(PROJECT_ROOT)

    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    det_model = YOLO(args.weights)
    cam_model = YOLO(args.weights)
    net = cam_model.model.to(device)
    net.eval()
    net.requires_grad_(True)

    if args.layer < 0 or args.layer >= len(net.model):
        raise IndexError(f"layer 超出范围，当前模型共有 {len(net.model)} 层。")

    hook = ActivationsAndGradients(net.model[args.layer])
    image_paths = collect_images(args.source, args.max_images)

    for image_path in image_paths:
        selected = select_detection(det_model, image_path, args.imgsz, args.conf, args.device, args.det_index)
        if selected is None:
            print(f"[SKIP] {image_path.name} 没有检测框")
            continue

        resized_bgr, tensor, gain, pad_w, pad_h = preprocess(image_path, args.imgsz, device)
        target_box = convert_box_to_letterbox(selected["box"], gain, pad_w, pad_h, device)

        net.zero_grad(set_to_none=True)
        detect_head = net.model[-1]
        detect_training = detect_head.training
        detect_head.train()
        preds = net(tensor)
        detect_head.train(detect_training)

        raw_scores = preds["one2many"]["scores"]
        decoded_boxes = detect_head._get_decode_boxes(preds["one2many"]).permute(0, 2, 1)[0]
        cls_scores = raw_scores[0, selected["cls"]].sigmoid()

        match_score = box_iou(decoded_boxes.detach(), target_box) * cls_scores.detach()
        anchor_idx = int(match_score.argmax().item())
        target = raw_scores[0, selected["cls"], anchor_idx]
        target.backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("未成功捕获特征图或梯度，请尝试更换 --layer。")

        cam = make_gradcam(hook.activations, hook.gradients, resized_bgr.shape[:2])
        overlay = overlay_heatmap(resized_bgr, cam, args.alpha)
        if args.draw_box:
            draw_box(overlay, target_box.detach().cpu().numpy(), f"cls={selected['cls']} conf={selected['conf']:.3f}")

        stem = image_path.stem
        suffix = f"det{selected['det_index']}_cls{selected['cls']}"
        heatmap_path = output_dir / f"{stem}_{suffix}_gradcam.jpg"
        rawcam_path = output_dir / f"{stem}_{suffix}_cam.jpg"

        cv2.imwrite(str(heatmap_path), overlay)
        cv2.imwrite(str(rawcam_path), np.uint8(cam * 255))

        print(f"[OK] {image_path.name}")
        print(f"     save      : {heatmap_path}")
        print(f"     det index : {selected['det_index']}")
        print(f"     class/conf: {selected['cls']} / {selected['conf']:.3f}")
        print(f"     anchor idx: {anchor_idx}")

    hook.remove()


if __name__ == "__main__":
    main()

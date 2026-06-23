"""Shared helpers for TorchVision detectors (Faster R-CNN).

Phần này giữ mọi thứ mà cả `train_torchvision.py` và `evaluate_torchvision.py`
dùng chung, để hai script đó mỏng và dễ đọc:

1. `YoloDetectionDataset`  : đọc dataset YOLO (cùng format BDD/DAWN đã chuẩn bị)
   và trả về target đúng chuẩn TorchVision (boxes xyxy pixel + labels 1..C).
2. `build_fasterrcnn`      : tạo Faster R-CNN COCO pretrained, thay head cho
   `num_classes` (gồm cả background = lớp 0).
3. `evaluate_detector`     : tính mAP50 / mAP50-95 (torchmetrics) + Precision /
   Recall tại một ngưỡng cố định để điền bảng so sánh.
4. `benchmark_batch1`      : đo latency batch-1 giống evaluate.py của YOLO.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_iou
from torchvision.transforms.v2 import functional as F

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------


def resolve_device(device: Any) -> torch.device:
    """Map a config device value ("0", "cpu", "cuda:0") to a torch.device."""
    value = str(device)
    if value == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


# ---------------------------------------------------------------------------
# DATASET
# YOLO labels are normalized [class cx cy w h]; Faster R-CNN wants absolute
# [x1 y1 x2 y2] pixel boxes and integer labels where 0 is background.
# ---------------------------------------------------------------------------


def _label_for_image(image_path: Path) -> Path:
    """Mirror the Ultralytics convention: .../images/.../x.jpg -> .../labels/.../x.txt."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _resolve_split_dir(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_value = data.get(split)
    if not split_value:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")

    split_dir = Path(split_value)
    if not split_dir.is_absolute():
        split_dir = (root / split_dir).resolve()
    return split_dir


class YoloDetectionDataset(Dataset):
    """Read a YOLO-format split and yield (image_tensor, target) for TorchVision."""

    def __init__(
        self,
        data_yaml: Path | None = None,
        split: str | None = None,
        images: list[Path] | None = None,
    ):
        # Either scan a YOLO split directory, or use an explicit image list
        # (used by per-weather evaluation to keep only one weather's images).
        if images is not None:
            self.images = sorted(Path(path) for path in images)
        else:
            images_dir = _resolve_split_dir(data_yaml, split)
            self.images = sorted(
                path for path in images_dir.rglob("*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
        if not self.images:
            raise FileNotFoundError("No images found for this dataset/split.")

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, image_path: Path) -> Path:
        return _label_for_image(image_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.images[index]
        image = read_image(str(image_path))  # uint8 CHW
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        image = F.to_dtype(image, torch.float32, scale=True)  # -> [0, 1]

        _, height, width = image.shape
        boxes: list[list[float]] = []
        labels: list[int] = []

        label_path = self._label_path(image_path)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = (float(value) for value in parts)
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
                y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # +1: reserve 0 for background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return image, target


def collate_fn(batch: list[tuple]) -> tuple[list, list]:
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------


def build_fasterrcnn(num_classes: int, imgsz: int, coco_pretrained: bool) -> torch.nn.Module:
    """Faster R-CNN ResNet50-FPN with the box head resized to `num_classes`."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if coco_pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=imgsz,
        max_size=imgsz,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# EVALUATION
# mAP50 / mAP50-95 from torchmetrics (COCO-style, comparable to Ultralytics).
# Precision / Recall via a fixed-threshold greedy match for the report table.
# ---------------------------------------------------------------------------


@torch.no_grad()
def precision_recall_at(
    predictions: list[dict],
    targets: list[dict],
    conf: float,
    iou_threshold: float,
) -> tuple[float, float]:
    """Greedy class-aware matching at a single (conf, IoU) operating point."""
    tp = fp = fn = 0
    for prediction, target in zip(predictions, targets):
        keep = prediction["scores"] >= conf
        pred_boxes = prediction["boxes"][keep]
        pred_labels = prediction["labels"][keep]
        order = prediction["scores"][keep].argsort(descending=True)

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for index in order.tolist():
            if len(gt_boxes) == 0:
                fp += 1
                continue
            ious = box_iou(pred_boxes[index : index + 1], gt_boxes)[0]
            ious = ious.clone()
            ious[gt_labels != pred_labels[index]] = 0
            ious[matched] = 0
            best_iou, best_gt = (ious.max(0) if len(ious) else (torch.tensor(0.0), None))
            if best_gt is not None and best_iou >= iou_threshold:
                matched[best_gt] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


@torch.no_grad()
def evaluate_detector(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    pr_conf: float = 0.25,
    pr_iou: float = 0.5,
) -> dict[str, Any]:
    """Run the model over `loader` and return the standard metric dictionary."""
    from torchmetrics.detection import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    all_predictions: list[dict] = []
    all_targets: list[dict] = []

    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        cpu_outputs = [{key: value.cpu() for key, value in output.items()} for output in outputs]
        cpu_targets = [
            {"boxes": target["boxes"], "labels": target["labels"]} for target in targets
        ]
        metric.update(cpu_outputs, cpu_targets)
        all_predictions.extend(cpu_outputs)
        all_targets.extend(cpu_targets)

    summary = metric.compute()
    precision, recall = precision_recall_at(all_predictions, all_targets, pr_conf, pr_iou)

    per_class_ap = {}
    if "map_per_class" in summary and summary["map_per_class"].ndim > 0:
        classes = summary.get("classes")
        for class_id, ap in zip(classes.tolist(), summary["map_per_class"].tolist()):
            per_class_ap[int(class_id)] = float(ap)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "map50": float(summary["map_50"]),
        "map50_95": float(summary["map"]),
        "mar_100": float(summary["mar_100"]),
        "per_class_map50_95": per_class_ap,
        "pr_operating_point": {"conf": pr_conf, "iou": pr_iou},
    }


@torch.no_grad()
def benchmark_batch1(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float | None, float | None]:
    """Single-image latency to match the YOLO evaluation protocol."""
    if image is None:
        return None, None
    model.eval()
    sample = [image.to(device)]

    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000 / iterations
    return latency_ms, 1000 / latency_ms

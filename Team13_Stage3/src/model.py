"""Stage 3 - detector definition.

Detector  : YOLOv8s (Ultralytics, one-stage anchor-free)
Backbone  : CSPDarknet-53 (small width/depth scaling)
Pretrained: COCO detection weights shipped by Ultralytics (``yolov8s.pt``).
            They are a *generic* COCO detector - no checkpoint pretrained or
            fine-tuned on NEU-DET / FactoryVision is used anywhere.

Why a one-stage anchor-free detector for this task
--------------------------------------------------
* The 200x200 steel crops contain few, large, low-contrast objects; a heavy
  two-stage proposal network buys little and costs a lot of latency.
* An anchor-free head avoids hand-tuning anchor sizes for the very elongated
  ``scratches`` boxes and the near-full-image ``crazing`` boxes.
* A single forward pass + NMS stays well inside a production line's cycle
  time on a single mid-range GPU.

``load_detector`` is the only place a checkpoint is ever opened, so training,
evaluation and the benchmark script share exactly the same loading path.
"""
import os

DEFAULT_PRETRAINED = "yolov8s.pt"
BACKBONE = "CSPDarknet (YOLOv8s)"
PRETRAINED_SOURCE = "Ultralytics YOLOv8s COCO-pretrained weights (yolov8s.pt)"


def load_detector(weights):
    """Load an Ultralytics YOLO model from a local .pt file."""
    from ultralytics import YOLO
    return YOLO(weights)


def build_for_training(pretrained=DEFAULT_PRETRAINED):
    """Model used for fine-tuning on the FactoryVision detection split."""
    return load_detector(pretrained)


def model_info(model):
    """Parameter count / layer count of a loaded Ultralytics model."""
    m = model.model
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"params": total, "trainable_params": trainable,
            "layers": len(list(m.modules()))}


def checkpoint_size_mb(path):
    return os.path.getsize(path) / 1024 ** 2

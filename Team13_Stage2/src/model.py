"""Stage 2 - pretrained architectures and fine-tuning strategies.

Architecture families (two different families, as required):
    * ResNet18      (residual family)
    * DenseNet121   (dense-connectivity family)
    * MobileNetV3-S (inverted-residual family, extra deployment reference)

Fine-tuning strategies:
    * feature_extraction : backbone frozen, only the new classifier is trained
    * full_finetune      : every parameter is trained
    * last_k             : only the last k top-level backbone stages + head
                           are trainable (k is recorded in the config)
    * l2sp               : full fine-tuning + L2-SP regularization towards the
                           pretrained weights (coefficient recorded)
"""
import torch
import torch.nn as nn
from torchvision import models

from common import NUM_CLASSES

ARCHITECTURES = ("resnet18", "densenet121", "mobilenet_v3_small")
STRATEGIES = ("feature_extraction", "full_finetune", "last_k", "l2sp")


def build_backbone(arch, pretrained=True):
    """Return (model, name_of_classifier_attribute)."""
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m, "fc"
    if arch == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
        return m, "classifier"
    if arch == "mobilenet_v3_small":
        w = (models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
             if pretrained else None)
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
        return m, "classifier"
    raise ValueError("unknown architecture: " + arch)


def _top_level_stages(arch, model):
    """Ordered list of (name, module) used by the last-k strategy."""
    if arch == "resnet18":
        names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        return [(n, getattr(model, n)) for n in names]
    if arch == "densenet121":
        f = model.features
        names = ["conv0", "norm0", "denseblock1", "transition1",
                 "denseblock2", "transition2", "denseblock3", "transition3",
                 "denseblock4", "norm5"]
        return [(n, getattr(f, n)) for n in names]
    if arch == "mobilenet_v3_small":
        return [(str(i), b) for i, b in enumerate(model.features)]
    raise ValueError(arch)


def apply_strategy(model, arch, head_attr, strategy, last_k=2):
    """Freeze / unfreeze parameters and return a description dict."""
    for p in model.parameters():
        p.requires_grad = True
    head = getattr(model, head_attr)
    frozen, trainable = [], []

    if strategy == "feature_extraction":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(head_attr)
        frozen = ["entire backbone (all layers before the classifier)"]
        trainable = [head_attr + " (new classifier head)"]

    elif strategy in ("full_finetune", "l2sp"):
        trainable = ["all layers (backbone + classifier)"]

    elif strategy == "last_k":
        stages = _top_level_stages(arch, model)
        keep = set(id(m) for _, m in stages[-last_k:])
        for name, mod in stages:
            req = id(mod) in keep
            for p in mod.parameters():
                p.requires_grad = req
            (trainable if req else frozen).append(name)
        for p in head.parameters():
            p.requires_grad = True
        trainable.append(head_attr + " (classifier head)")
    else:
        raise ValueError("unknown strategy: " + strategy)

    return {"strategy": strategy, "last_k": last_k if strategy == "last_k"
            else None, "frozen_layers": frozen, "trainable_layers": trainable}


def build(arch, strategy, pretrained=True, last_k=2):
    model, head_attr = build_backbone(arch, pretrained)
    info = apply_strategy(model, arch, head_attr, strategy, last_k)
    info["architecture"] = arch
    info["head_attr"] = head_attr
    return model, info


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class L2SP:
    """L2-SP regularizer: ||w - w_pretrained||^2 on the backbone, ||w||^2 on
    the newly initialised head (Xuhong et al., ICML 2018)."""

    def __init__(self, model, head_attr, alpha=0.01, beta=0.01):
        self.alpha, self.beta, self.head_attr = alpha, beta, head_attr
        self.ref = {n: p.detach().clone()
                    for n, p in model.named_parameters()
                    if not n.startswith(head_attr)}

    def to(self, device):
        self.ref = {n: v.to(device) for n, v in self.ref.items()}
        return self

    def __call__(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.ref:
                loss = loss + self.alpha * 0.5 * ((p - self.ref[n]) ** 2).sum()
            else:
                loss = loss + self.beta * 0.5 * (p ** 2).sum()
        return loss


if __name__ == "__main__":
    for arch in ARCHITECTURES:
        for st in ("feature_extraction", "full_finetune", "last_k"):
            m, info = build(arch, st, pretrained=False)
            t, tr = count_parameters(m)
            out = m(torch.zeros(1, 3, 224, 224))
            print("{:20s} {:18s} total={:>10,} trainable={:>10,} out={}"
                  .format(arch, st, t, tr, tuple(out.shape)))

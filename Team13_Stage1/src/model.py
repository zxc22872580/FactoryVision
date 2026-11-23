"""Stage 1 - self-designed baseline CNN (trained fully from scratch).

The architecture deliberately uses only generic building blocks
(Convolution -> Activation -> Pooling -> Classifier); it is NOT a ResNet /
VGG / MobileNet variant and no pretrained weight is ever loaded.

``use_bn`` switches Improvement Method B (Batch Normalization) on/off so that
the exact same code path serves all four ablation experiments.
"""
import torch
import torch.nn as nn

from common import NUM_CLASSES


def conv_block(cin, cout, use_bn):
    layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU(inplace=True))
    return layers


class BaselineCNN(nn.Module):
    """4-stage plain CNN:  (Conv-[BN]-ReLU) x2 -> MaxPool, then a MLP head."""

    def __init__(self, num_classes=NUM_CLASSES, in_ch=1, use_bn=False,
                 dropout=0.0, width=(32, 64, 128, 256)):
        super().__init__()
        self.use_bn = use_bn
        self.dropout_p = dropout
        c1, c2, c3, c4 = width
        feats = []
        feats += conv_block(in_ch, c1, use_bn) + conv_block(c1, c1, use_bn)
        feats += [nn.MaxPool2d(2)]                       # 128 -> 64
        feats += conv_block(c1, c2, use_bn) + conv_block(c2, c2, use_bn)
        feats += [nn.MaxPool2d(2)]                       # 64 -> 32
        feats += conv_block(c2, c3, use_bn) + conv_block(c3, c3, use_bn)
        feats += [nn.MaxPool2d(2)]                       # 32 -> 16
        feats += conv_block(c3, c4, use_bn)
        feats += [nn.MaxPool2d(2)]                       # 16 -> 8
        self.features = nn.Sequential(*feats)
        self.pool = nn.AdaptiveAvgPool2d(1)
        head = [nn.Flatten(), nn.Linear(c4, 128), nn.ReLU(inplace=True)]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*head)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(use_bn=False, dropout=0.0, in_ch=1, num_classes=NUM_CLASSES):
    return BaselineCNN(num_classes=num_classes, in_ch=in_ch, use_bn=use_bn,
                       dropout=dropout)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for bn in (False, True):
        m = build_model(use_bn=bn)
        t, tr = count_parameters(m)
        y = m(torch.zeros(2, 1, 128, 128))
        print(f"use_bn={bn}: params={t:,} trainable={tr:,} out={tuple(y.shape)}")

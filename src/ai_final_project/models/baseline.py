from __future__ import annotations

from torchvision.models import resnet18


def build_cifar_resnet18(num_classes: int = 10):
    model = resnet18(num_classes=num_classes)
    model.conv1 = model.conv1.__class__(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = model.maxpool.__class__(kernel_size=1, stride=1, padding=0)
    return model

"""Teacher network: ResNet-50 adapted to 32x32 CIFAR-100 images.

The standard ImageNet ResNet-50 starts with a 7x7 stride-2 convolution followed by a stride-2
max-pool, which shrinks a 32x32 CIFAR image to 8x8 before the first residual stage and throws
away most of the spatial detail. The usual CIFAR adaptation (He et al. 2016, and nearly every
CIFAR codebase) is used here instead:

    * stem  : 3x3 convolution, stride 1, 64 channels   (replaces 7x7 / stride 2)
    * no max-pool after the stem
    * stages: the four standard bottleneck stages [3, 4, 6, 3] -> feature maps 32, 16, 8, 4
    * head  : global average pool + Linear(2048 -> 100)

All bottleneck stages are initialised from the ImageNet-1k pretrained torchvision weights
(IMAGENET1K_V2) and then fine-tuned end-to-end on CIFAR-100; the new stem and the new classifier
start from random initialisation. Starting from ImageNet features gives a much stronger teacher
in a small training budget, which is exactly what knowledge distillation needs.

For feature-based distillation the network also returns its intermediate representations.
The default hint layer is the output of `layer3` (1024 channels at 8x8), the same spatial size
as the hint layers chosen for the three students.
"""
import torch
import torch.nn as nn
import torchvision


class ResNet50Teacher(nn.Module):
    # name of the intermediate feature map that student hints are matched to
    hint_layer = "layer3"
    hint_channels = 1024

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)

        # CIFAR stem: 3x3 / stride 1, no max-pool (keeps 32x32 resolution into layer1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.bn1 = net.bn1
        self.relu = nn.ReLU(inplace=True)

        # pretrained bottleneck stages (torchvision layout, weights reused unchanged)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_features: bool = False):
        x = self.relu(self.bn1(self.conv1(x)))          # 64  x 32 x 32
        f1 = self.layer1(x)                              # 256 x 32 x 32
        f2 = self.layer2(f1)                             # 512 x 16 x 16
        f3 = self.layer3(f2)                             # 1024 x 8 x 8   <- hint layer
        f4 = self.layer4(f3)                             # 2048 x 4 x 4
        logits = self.fc(torch.flatten(self.avgpool(f4), 1))
        if return_features:
            return logits, {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}
        return logits

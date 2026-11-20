"""Three lightweight student networks, implemented from scratch and adapted to 32x32 inputs.

All three follow their original papers block for block. The only changes are the standard
CIFAR adaptations: early strides are reduced so that a 32x32 image is not shrunk to 1x1 before
the classifier (ImageNet models downsample by 32x, which would leave 1x1 feature maps here).

    Model              Paper                                   Downsampling here   Hint layer (8x8)
    MobileNetV2        Sandler et al., CVPR 2018               32 -> 16 -> 8 -> 4   output of the 96-ch stage
    SqueezeNet 1.1     Iandola et al., 2016 (v1.1 variant)     32 -> 16 -> 8 -> 4   output of fire5 (256 ch)
    EfficientNet-Lite0 Tan & Le, ICML 2019 + TF "Lite" variant 32 -> 16 -> 8 -> 4   output of the 112-ch stage

Every student exposes the same interface:

    logits = model(x)
    logits, hint = model(x, return_features=True)   # hint: (B, C_hint, 8, 8) feature map

`hint` is the intermediate representation used for feature-based (FitNets-style) distillation;
`hint_channels` tells the distillation framework how wide it is.
"""
import math

import torch
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    """Round channel counts to a multiple of 8 (same helper as the reference implementations)."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:          # do not round down by more than 10 %
        new_v += divisor
    return new_v


def _init_weights(model: nn.Module):
    """Kaiming init for convolutions, small normal init for the classifier (as in torchvision)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm2d -> activation (the basic unit of MobileNet/EfficientNet)."""

    def __init__(self, cin, cout, k=3, stride=1, groups=1, act=nn.ReLU6):
        layers = [nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act is not None:
            layers.append(act(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 / EfficientNet-Lite MBConv block (no squeeze-excitation).

    1x1 expansion (skipped if expand_ratio == 1) -> kxk depthwise -> 1x1 linear projection.
    A residual connection is used when the block keeps resolution and width.
    """

    def __init__(self, cin, cout, stride, expand_ratio, kernel=3, act=nn.ReLU6):
        super().__init__()
        hidden = int(round(cin * expand_ratio))
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(cin, hidden, 1, act=act))                    # expansion
        layers += [ConvBNAct(hidden, hidden, kernel, stride, groups=hidden, act=act),  # depthwise
                   nn.Conv2d(hidden, cout, 1, bias=False),                       # projection
                   nn.BatchNorm2d(cout)]                                         # (linear, no act)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


# ============================================================================ MobileNetV2
class MobileNetV2(nn.Module):
    """MobileNetV2 (width 1.0) for 32x32 inputs.

    Inverted-residual settings (t = expansion, c = channels, n = repeats, s = stride) follow
    Table 2 of the paper. CIFAR adaptation: the stem and the second stage use stride 1, so the
    total downsampling is 8x (32 -> 4) instead of 32x.
    """
    hint_channels = 96

    # t, c, n, s       (paper strides: 2 | 1, 2, 2, 2, 1, 2, 1)
    SETTINGS = [[1, 16, 1, 1],
                [6, 24, 2, 1],      # stride 2 -> 1 for CIFAR
                [6, 32, 3, 2],      # 32 -> 16
                [6, 64, 4, 2],      # 16 -> 8
                [6, 96, 3, 1],      # 8x8  <- hint layer is the output of this stage
                [6, 160, 3, 2],     # 8 -> 4
                [6, 320, 1, 1]]

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        cin = _make_divisible(32 * width_mult)
        self.last_channels = _make_divisible(1280 * max(1.0, width_mult))
        self.stem = ConvBNAct(3, cin, 3, stride=1)                   # stride 2 -> 1 for CIFAR
        stages = []
        for t, c, n, s in self.SETTINGS:
            cout = _make_divisible(c * width_mult)
            blocks = [InvertedResidual(cin if i == 0 else cout, cout, s if i == 0 else 1, t)
                      for i in range(n)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.hint_stage = 4                                          # index of the 96-ch stage
        self.head = ConvBNAct(cin, self.last_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_channels, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        hint = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.hint_stage:
                hint = x
        x = self.pool(self.head(x)).flatten(1)
        logits = self.classifier(x)
        return (logits, hint) if return_features else logits


# ============================================================================ SqueezeNet
class Fire(nn.Module):
    """SqueezeNet Fire module: 1x1 squeeze, then parallel 1x1 and 3x3 expand, concatenated.

    BatchNorm is added after every convolution. The original SqueezeNet has no BatchNorm and is
    notoriously hard to train from scratch at the learning rates used here; BN is the standard
    fix in CIFAR re-implementations and does not change the architecture's topology or its
    parameter count meaningfully (BN layers can be folded into the convolutions at export).
    """

    def __init__(self, cin, squeeze, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1, bias=False),
                                     nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.expand1x1 = nn.Sequential(nn.Conv2d(squeeze, expand1x1, 1, bias=False),
                                       nn.BatchNorm2d(expand1x1), nn.ReLU(inplace=True))
        self.expand3x3 = nn.Sequential(nn.Conv2d(squeeze, expand3x3, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(expand3x3), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


class SqueezeNet(nn.Module):
    """SqueezeNet 1.1 for 32x32 inputs.

    v1.1 layout: conv1 -> pool -> fire2, fire3 -> pool -> fire4, fire5 -> pool -> fire6..fire9
    -> dropout -> 1x1 conv classifier -> global average pool.
    CIFAR adaptation: conv1 is 3x3 / stride 1 (instead of 3x3 / stride 2), so the three
    max-pools give 32 -> 16 -> 8 -> 4.
    """
    hint_channels = 256

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        pool = lambda: nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.block1 = nn.Sequential(pool(), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64))       # 16x16
        self.block2 = nn.Sequential(pool(), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))  # 8x8 <- hint
        self.block3 = nn.Sequential(pool(), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                                    Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))          # 4x4
        # classifier: 1x1 conv to class scores, then global average pooling (no FC layer).
        # The original applies a ReLU to the class scores; it is dropped here because clipping
        # all negative logits to 0 erases the ranking among unlikely classes, which is precisely
        # the "dark knowledge" that soft-target distillation transfers.
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                                        nn.AdaptiveAvgPool2d(1))
        _init_weights(self)
        # the final conv uses a small init as in torchvision's SqueezeNet
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.01)

    def forward(self, x, return_features=False):
        x = self.block1(self.stem(x))
        hint = self.block2(x)
        logits = self.classifier(self.block3(hint)).flatten(1)
        return (logits, hint) if return_features else logits


# ============================================================================ EfficientNet-Lite0
class EfficientNetLite0(nn.Module):
    """EfficientNet-Lite0 for 32x32 inputs.

    EfficientNet-Lite is the edge-friendly variant of EfficientNet published with the TensorFlow
    Lite model zoo: compared with EfficientNet-B0 it (1) removes squeeze-and-excitation blocks,
    (2) replaces Swish with ReLU6 (quantisation friendly), and (3) keeps the stem (32) and head
    (1280) widths fixed. Lite0 uses width = depth = 1.0, i.e. the B0 block table below.
    CIFAR adaptation: stem stride 2 -> 1 and stage-2 stride 2 -> 1 (total downsampling 8x).
    """
    hint_channels = 112

    # expand, kernel, stride, out_channels, repeats        (paper strides: 2 | 1,2,2,2,1,2,1)
    BLOCKS = [[1, 3, 1, 16, 1],
              [6, 3, 1, 24, 2],     # stride 2 -> 1 for CIFAR
              [6, 5, 2, 40, 2],     # 32 -> 16
              [6, 3, 2, 80, 3],     # 16 -> 8
              [6, 5, 1, 112, 3],    # 8x8  <- hint layer
              [6, 5, 2, 192, 4],    # 8 -> 4
              [6, 3, 1, 320, 1]]

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = ConvBNAct(3, 32, 3, stride=1)                    # stride 2 -> 1 for CIFAR
        cin, stages = 32, []
        for e, k, s, c, n in self.BLOCKS:
            blocks = [InvertedResidual(cin if i == 0 else c, c, s if i == 0 else 1, e, kernel=k)
                      for i in range(n)]
            stages.append(nn.Sequential(*blocks))
            cin = c
        self.stages = nn.ModuleList(stages)
        self.hint_stage = 4
        self.head = ConvBNAct(cin, 1280, 1)                          # fixed head width (Lite)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        _init_weights(self)
        # the classifier of EfficientNet uses a uniform init scaled by 1/sqrt(fan_out)
        r = 1.0 / math.sqrt(num_classes)
        nn.init.uniform_(self.classifier[1].weight, -r, r)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        hint = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.hint_stage:
                hint = x
        x = self.pool(self.head(x)).flatten(1)
        logits = self.classifier(x)
        return (logits, hint) if return_features else logits

"""Automated architecture generator (Question 2, requirement 2).

`build_network(config)` turns an architecture config (see search_space.py) into a ready-to-train
PyTorch module. Every network has the same overall template, and the config decides what goes
into each slot:

    input (R x R) -> stem 3x3 conv -> stage 0 -> stage 1 (stride 2) -> ... -> 1x1 head conv
                  -> global average pool -> dropout -> linear classifier

The generated module stores its own config (`model.config`), so a checkpoint can always be
rebuilt exactly, and `model.input_resolution` tells the inference code what image size to feed.
"""
import torch
import torch.nn as nn

from .search_space import validate_config

ACTIVATION_LAYERS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "hardswish": lambda: nn.Hardswish(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "leakyrelu": lambda: nn.LeakyReLU(0.1, inplace=True),
}


def make_act(name: str) -> nn.Module:
    return ACTIVATION_LAYERS[name]()


def conv_bn(cin, cout, k, stride=1, groups=1, act=None):
    """kxk convolution (same padding) + BatchNorm (+ optional activation)."""
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act is not None:
        layers.append(make_act(act))
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Standard convolution block: kxk conv -> BN -> act, with an identity shortcut when the
    input and output shapes are equal (ResNet-style), which keeps deeper candidates trainable."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = conv_bn(cin, cout, k, stride, act=act)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        out = self.body(x)
        return out + x if self.residual else out


class DWSeparableBlock(nn.Module):
    """Depthwise-separable convolution (MobileNetV1): kxk depthwise conv -> BN -> act ->
    1x1 pointwise conv -> BN -> act. Costs about k^2 + C_out multiply-adds per input channel
    instead of k^2 * C_out for a standard convolution."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.depthwise = conv_bn(cin, cin, k, stride, groups=cin, act=act)
        self.pointwise = conv_bn(cin, cout, 1, act=act)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        out = self.pointwise(self.depthwise(x))
        return out + x if self.residual else out


class MBConvBlock(nn.Module):
    """Inverted residual / MBConv (MobileNetV2): 1x1 expansion -> kxk depthwise -> 1x1 linear
    projection (no activation after the projection), residual when shapes match."""

    def __init__(self, cin, cout, k, stride, act, expand):
        super().__init__()
        hidden = cin * expand
        self.body = nn.Sequential(conv_bn(cin, hidden, 1, act=act),
                                  conv_bn(hidden, hidden, k, stride, groups=hidden, act=act),
                                  conv_bn(hidden, cout, 1, act=None))
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        out = self.body(x)
        return out + x if self.residual else out


def make_block(kind, cin, cout, k, stride, act, expand):
    if kind == "conv":
        return ConvBlock(cin, cout, k, stride, act)
    if kind == "dwsep":
        return DWSeparableBlock(cin, cout, k, stride, act)
    if kind == "mbconv":
        return MBConvBlock(cin, cout, k, stride, act, expand)
    raise ValueError(f"unknown block type {kind}")


class GeneratedCNN(nn.Module):
    """A CNN assembled from an architecture config."""

    def __init__(self, config: dict, num_classes: int = 10):
        super().__init__()
        validate_config(config)
        self.config = config
        self.input_resolution = config["resolution"]
        act = config["activation"]
        cin = config["stem_channels"]
        self.stem = conv_bn(3, cin, 3, 1, act=act)

        stages = []
        for s, st in enumerate(config["stages"]):
            blocks = []
            for b in range(st["depth"]):
                stride = 2 if (s > 0 and b == 0) else 1      # downsample at the start of stages 1..n
                blocks.append(make_block(st["block"], cin, st["width"], st["kernel"], stride, act,
                                         st.get("expand", 1)))
                cin = st["width"]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        head = max(cin, 128)                                  # small 1x1 head before pooling
        self.head = conv_bn(cin, head, 1, act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(config.get("dropout", 0.0)), nn.Linear(head, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(self.stages(self.stem(x)))
        return self.classifier(self.pool(x).flatten(1))


def build_network(config: dict, num_classes: int = 10) -> GeneratedCNN:
    """Public entry point of the generator."""
    return GeneratedCNN(config, num_classes)

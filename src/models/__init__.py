"""Model registry: build any teacher / student network by name."""
from .students import EfficientNetLite0, MobileNetV2, SqueezeNet
from .teacher import ResNet50Teacher

STUDENTS = {
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}

# readable names for tables and plots
DISPLAY_NAMES = {
    "resnet50": "ResNet-50 (teacher)",
    "mobilenetv2": "MobileNetV2",
    "squeezenet": "SqueezeNet 1.1",
    "efficientnet_lite0": "EfficientNet-Lite0",
}


def build_student(name: str, num_classes: int = 100):
    """Create an untrained student network by its registry name."""
    if name not in STUDENTS:
        raise ValueError(f"unknown student '{name}', choose from {list(STUDENTS)}")
    return STUDENTS[name](num_classes=num_classes)


def build_teacher(num_classes: int = 100, pretrained: bool = True):
    return ResNet50Teacher(num_classes=num_classes, pretrained=pretrained)

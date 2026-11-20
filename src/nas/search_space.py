"""Searchable CNN architecture space for the hardware-aware NAS (Question 2, requirement 1).

An architecture is described by a plain, JSON-serialisable dictionary ("config"):

    {
      "resolution": 32,                     # input image size  (variable input resolution)
      "activation": "hardswish",            # activation used everywhere (different activations)
      "stem_channels": 24,                  # width of the 3x3 stem convolution
      "stages": [                           # 3 or 4 stages            (variable depth)
        {"block": "dwsep",                  # conv | dwsep | mbconv    (depthwise-separable blocks)
         "depth": 2,                        # blocks in this stage     (variable depth)
         "width": 32,                       # output channels          (variable filters)
         "kernel": 5,                       # 3 | 5 | 7                (different kernel sizes)
         "expand": 1},                      # expansion ratio (only used by mbconv)
        ...
      ],
      "dropout": 0.1
    }

Stage 0 keeps the resolution; every later stage starts with a stride-2 block, so a 4-stage
network at 32x32 ends at 4x4 and a 3-stage network at 8x8.

Block types
  conv   : standard kxk convolution + BN + activation (ResNet-style residual when shapes match)
  dwsep  : depthwise-separable convolution (MobileNetV1): kxk depthwise + 1x1 pointwise
  mbconv : inverted residual (MobileNetV2): 1x1 expand -> kxk depthwise -> 1x1 linear projection

The same space can be sampled in two ways:
  * `suggest_config(trial)`  - by an Optuna trial (Bayesian optimisation with TPE)
  * `random_config(rng)`     - uniformly at random (used to calibrate the deployment constraints)
"""
import hashlib
import json
import random

# ----------------------------------------------------------------------------- the search space
RESOLUTIONS = [24, 32, 40]
ACTIVATIONS = ["relu", "relu6", "hardswish", "silu", "leakyrelu"]
STEM_CHANNELS = [16, 24, 32]
NUM_STAGES = [3, 4]
BLOCK_TYPES = ["conv", "dwsep", "mbconv"]
DEPTHS = [1, 2, 3]
KERNELS = [3, 5, 7]
EXPANSIONS = [2, 4, 6]
DROPOUTS = [0.0, 0.1, 0.2]
# channel choices grow with depth, as in hand-designed CNNs (wider when the map is smaller)
STAGE_WIDTHS = [
    [16, 24, 32, 48],        # stage 0 (full resolution)
    [32, 48, 64, 96],        # stage 1 (1/2)
    [64, 96, 128, 160],      # stage 2 (1/4)
    [96, 128, 192, 256],     # stage 3 (1/8)
]


def search_space_description() -> dict:
    """The full space as a dict (saved next to the NAS results and quoted in the report)."""
    return {"resolution": RESOLUTIONS, "activation": ACTIVATIONS, "stem_channels": STEM_CHANNELS,
            "num_stages": NUM_STAGES, "block": BLOCK_TYPES, "depth_per_stage": DEPTHS,
            "kernel": KERNELS, "mbconv_expansion": EXPANSIONS, "dropout": DROPOUTS,
            "stage_widths": STAGE_WIDTHS}


def search_space_size() -> int:
    """Number of distinct architectures in the space (for the report)."""
    per_stage = lambda: len(DEPTHS) * len(KERNELS) * (2 + len(EXPANSIONS))  # conv, dwsep, mbconv x expansions
    total = 0
    for n in NUM_STAGES:
        stages = 1
        for s in range(n):
            stages *= per_stage() * len(STAGE_WIDTHS[s])
        total += stages
    return total * len(RESOLUTIONS) * len(ACTIVATIONS) * len(STEM_CHANNELS) * len(DROPOUTS)


# ----------------------------------------------------------------------------- samplers
def suggest_config(trial) -> dict:
    """Sample one architecture with an Optuna trial (define-by-run, conditional parameters)."""
    cfg = {
        "resolution": trial.suggest_categorical("resolution", RESOLUTIONS),
        "activation": trial.suggest_categorical("activation", ACTIVATIONS),
        "stem_channels": trial.suggest_categorical("stem_channels", STEM_CHANNELS),
        "dropout": trial.suggest_categorical("dropout", DROPOUTS),
        "stages": [],
    }
    n_stages = trial.suggest_categorical("num_stages", NUM_STAGES)
    for s in range(n_stages):
        block = trial.suggest_categorical(f"s{s}_block", BLOCK_TYPES)
        stage = {"block": block,
                 "depth": trial.suggest_categorical(f"s{s}_depth", DEPTHS),
                 "width": trial.suggest_categorical(f"s{s}_width", STAGE_WIDTHS[s]),
                 "kernel": trial.suggest_categorical(f"s{s}_kernel", KERNELS),
                 # expansion only exists for inverted residual blocks (conditional parameter)
                 "expand": trial.suggest_categorical(f"s{s}_expand", EXPANSIONS) if block == "mbconv" else 1}
        cfg["stages"].append(stage)
    return cfg


def random_config(rng: random.Random) -> dict:
    """Sample one architecture uniformly at random from the same space."""
    cfg = {"resolution": rng.choice(RESOLUTIONS), "activation": rng.choice(ACTIVATIONS),
           "stem_channels": rng.choice(STEM_CHANNELS), "dropout": rng.choice(DROPOUTS), "stages": []}
    for s in range(rng.choice(NUM_STAGES)):
        block = rng.choice(BLOCK_TYPES)
        cfg["stages"].append({"block": block, "depth": rng.choice(DEPTHS),
                              "width": rng.choice(STAGE_WIDTHS[s]), "kernel": rng.choice(KERNELS),
                              "expand": rng.choice(EXPANSIONS) if block == "mbconv" else 1})
    return cfg


# ----------------------------------------------------------------------------- helpers
def validate_config(cfg: dict) -> None:
    """Raise ValueError if a config is outside the search space (guards hand-edited JSON files)."""
    if cfg["resolution"] not in RESOLUTIONS or cfg["activation"] not in ACTIVATIONS:
        raise ValueError("resolution/activation outside the search space")
    if cfg["stem_channels"] not in STEM_CHANNELS or len(cfg["stages"]) not in NUM_STAGES:
        raise ValueError("stem width / number of stages outside the search space")
    for s, st in enumerate(cfg["stages"]):
        if (st["block"] not in BLOCK_TYPES or st["depth"] not in DEPTHS
                or st["width"] not in STAGE_WIDTHS[s] or st["kernel"] not in KERNELS):
            raise ValueError(f"stage {s} outside the search space: {st}")


def config_id(cfg: dict) -> str:
    """Short stable hash of a config (identifies duplicates in the search history)."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def config_summary(cfg: dict) -> str:
    """One-line human readable description, e.g. 'r32 hardswish stem24 | dwsep k5 d2 w32 | ...'."""
    stages = " | ".join(
        f"{st['block']}{'x' + str(st['expand']) if st['block'] == 'mbconv' else ''} k{st['kernel']} "
        f"d{st['depth']} w{st['width']}" for st in cfg["stages"])
    return f"r{cfg['resolution']} {cfg['activation']} stem{cfg['stem_channels']} | {stages}"

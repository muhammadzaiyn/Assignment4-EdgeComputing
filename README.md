# Knowledge Distillation and Hardware-Aware NAS for Edge Deployment

Edge Computing & Intelligent Systems - Assignment 4.

| Question | What is built | Dataset |
|---|---|---|
| **Q1** Knowledge distillation | ResNet-50 teacher -> MobileNetV2, SqueezeNet 1.1, EfficientNet-Lite0 students, trained with and without KD (soft targets with temperature, hard + soft loss, feature hints), unified inference, ONNX Runtime CPU deployment | CIFAR-100 |
| **Q2** Hardware-aware NAS | searchable CNN space, architecture generator, Optuna TPE search with **measured** CPU latency / size / peak memory constraints, best model retrained from scratch, ONNX deployment | CIFAR-10 |

The technical report is [report/Technical_Report.pdf](report/Technical_Report.pdf) (editable copy:
`report/Technical_Report.docx`). Trained models are in `checkpoints/`, ONNX models in `onnx_models/`, and
all experimental results (CSV/JSON, figures, full console logs) in `results/`.

## Results at a glance

All numbers are FP32, on the official test sets, from `results/benchmark_summary.csv` and
`results/kd_comparison.csv`. Latency: batch 1, one CPU thread (AMD Ryzen 7 7700), median of 3 x 300 runs.

**Q1 - knowledge distillation on CIFAR-100** (teacher ResNet-50: 82.27% top-1, 23.7 M params, 28.2 ms)

| Student | Params | Baseline | KD soft only | KD soft + feature | Gain | PyTorch / ONNX Runtime latency |
|---|---|---|---|---|---|---|
| MobileNetV2 | 2.35 M | 73.88% | 76.92% | **78.68%** | +4.80 | 4.25 / 1.54 ms |
| SqueezeNet 1.1 | 0.78 M | 65.27% | 66.91% | **67.87%** | +2.60 | 2.02 / 0.34 ms |
| EfficientNet-Lite0 | 3.50 M | 74.20% | 76.41% | **78.96%** | +4.76 | 4.79 / 2.00 ms |

**Q2 - hardware-aware NAS on CIFAR-10** (constraints: CPU latency <= 2 ms, size <= 1 MB, peak memory <= 8 MB)

100 Optuna TPE trials: 36 rejected by the measured constraints, 52 pruned early, 11 trained to completion, 1 duplicate.
The selected architecture (32x32 input, hardswish, MBConv / depthwise-separable / MBConv stages) trained from scratch
reaches **91.10% test accuracy** with 0.159 M parameters, 0.68 MB, 1.6 MB peak memory and 1.08 ms
(PyTorch) / 0.37 ms (ONNX Runtime) CPU latency.

## Folder map

```
README.md                         this file
requirements.txt                  package versions used for the submitted results
run_all.py                        runs every step in order (--quick smoke test or --full)
src/
  common.py                       paths, seeding, logging (Tee), JSON/CSV helpers, system info
  data.py                         CIFAR-10/100 download + cache, 45k/5k/10k split, GPU augmentation loader
  models/teacher.py               ResNet-50 adapted to 32x32 (ImageNet-initialised)
  models/students.py              MobileNetV2, SqueezeNet 1.1, EfficientNet-Lite0 (from scratch)
  kd.py                           distillation framework: temperature softmax, soft-target KL,
                                  hint regressor + feature MSE, combined loss
  train_utils.py                  one training engine for teacher / baseline / KD / NAS runs
  metrics.py                      params, MACs, state-dict size, CPU latency, peak memory
  nas/search_space.py             searchable space + Optuna / random samplers
  nas/generator.py                builds a PyTorch network from an architecture config
  nas/hardware.py                 measured latency / size / memory, deployment constraints
  infer_utils.py                  load_model(name) for every trained model, ONNX Runtime session
  report_engine.py                renders the report blocks to PDF (reportlab) and DOCX
scripts/
  01_prepare_data.py              download datasets, verify the data pipeline
  02_train_teacher.py             Q1.1  teacher                           (deliverable 1)
  03_train_students_baseline.py   Q1.5  students without KD               (deliverable 3)
  04_train_students_kd.py         Q1.3-4 students with KD (+ --mode soft ablation)  (deliverable 4)
  05_nas_search.py                Q2    hardware-aware Optuna search      (deliverables 5-7)
  06_nas_train_best.py            Q2.7  train the selected architecture   (deliverable 8)
  07_export_onnx.py               Q1.7 / Q2.8 ONNX export + parity check  (deliverable 10)
  08_inference.py                 Q1.6  unified PyTorch inference         (deliverable 9)
  09_onnx_inference.py            ONNX Runtime CPU inference              (deliverable 10)
  10_benchmark.py                 automated benchmark -> CSV              (deliverable 11)
  11_make_figures.py              all figures + console captures
  12_make_report.py               builds the technical report             (deliverable 12)
  13_verify_deliverables.py       PASS/FAIL self-check of every deliverable
checkpoints/
  teacher/resnet50_cifar100.pth   baseline/*_baseline.pth   kd/*_kd.pth   kd/ablation/*_kd_soft.pth
  nas/nas_best.pth
onnx_models/                      6 students + NAS model (.onnx)
results/                          *.csv / *.json, students/, nas/, figures/, logs/
report/                           Technical_Report.pdf/.docx, screenshots/
```

## Requirements

* Python 3.11 or newer (results were produced with Python 3.14 on Windows 11).
* An NVIDIA GPU with a recent driver for training (used: RTX 5060 Ti 16 GB). Inference, ONNX Runtime and all
  latency measurements run on the CPU, so the trained models can be used without a GPU.
* About 6 GB of free disk space (PyTorch with CUDA ~3 GB, datasets ~0.7 GB with the downloaded archives).

## Installation

### Windows (PowerShell)

```powershell
cd edge_assignment4
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Linux (bash)

```bash
cd edge_assignment4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Pick the `cuXXX` wheel index that matches your driver (see pytorch.org). If `torch.cuda.is_available()` prints
`False`, training still runs on the CPU but is very slow.

## Running

Everything is run from the project root. The datasets are downloaded automatically on first use.

### One command

```powershell
python run_all.py --quick     # tiny version of every training step (a few minutes) to test the setup
python run_all.py --full      # every experiment as reported (about 3-4 hours on an RTX 5060 Ti)
```

`--quick` outputs carry a `_quick` suffix and never overwrite the real results.

### Step by step

```powershell
# 0. data
python scripts/01_prepare_data.py

# Q1 - knowledge distillation (CIFAR-100)
python scripts/02_train_teacher.py                       # ResNet-50 teacher, 40 epochs
python scripts/03_train_students_baseline.py             # 3 students without KD, 60 epochs each
python scripts/04_train_students_kd.py                   # 3 students with soft + feature KD
python scripts/04_train_students_kd.py --mode soft       # ablation: soft-target KD only
#   options: --arch mobilenetv2|squeezenet|efficientnet_lite0, --alpha, --temperature, --beta, --resume

# Q2 - hardware-aware NAS (CIFAR-10)
python scripts/05_nas_search.py --trials 100 --proxy-epochs 8 --fresh
#   automatically starts scripts/06_nas_train_best.py afterwards (use --no-train-best to skip)
#   constraints: --max-latency-ms 2.0 --max-size-mb 1.0 --max-peak-mem-mb 8.0

# deployment, inference, benchmark
python scripts/07_export_onnx.py
python scripts/08_inference.py --model all --num-samples 5
python scripts/08_inference.py --model mobilenetv2_kd --image path/to/picture.jpg
python scripts/08_inference.py --model all --evaluate
python scripts/09_onnx_inference.py --model all --num-samples 3
python scripts/09_onnx_inference.py --model all --evaluate
python scripts/10_benchmark.py

# figures, report, self-check
python scripts/11_make_figures.py
python scripts/12_make_report.py
python scripts/13_verify_deliverables.py
```

Model names accepted by the inference scripts: `mobilenetv2_baseline`, `mobilenetv2_kd`,
`squeezenet_baseline`, `squeezenet_kd`, `efficientnet_lite0_baseline`, `efficientnet_lite0_kd`, `nas_best`, or `all`.

## Where the outputs go

| Output | File |
|---|---|
| teacher / student training logs (per epoch) | `results/teacher_history.csv`, `results/students/*_history.csv` |
| final accuracies | `results/teacher_results.json`, `results/students/*.json` |
| KD comparison | `results/kd_comparison.csv` |
| NAS search history | `results/nas/search_history.csv` and `.json` |
| NAS best architecture | `results/nas/best_architecture.json`, `results/nas/final_results.json` |
| ONNX export checks | `results/onnx_export.csv` |
| benchmark | `results/benchmark_summary.csv`, `results/predictions_cifar100.csv`, `results/predictions_cifar10.csv` |
| figures | `results/figures/*.png` |
| full console logs | `results/logs/*.log` |

## Notes

* The ResNet-50 teacher checkpoint is about 95 MB, under GitHub's 100 MB per-file limit, so plain `git push`
  works (GitHub prints a warning for files over 50 MB). Git LFS is not required.
* `checkpoints/**/*_last.pth` are resume-only training states and are excluded by `.gitignore`.
* Latency numbers depend on the CPU and on background load; close other heavy programs before running the benchmark.

## Troubleshooting

* **Dataset download is slow** - the Toronto server is sometimes slow; `src/data.py` falls back to the Hugging Face
  mirror after about 3 minutes if the speed is too low (needs `pip install datasets`).
* **CUDA not available** - install the CUDA build of PyTorch from the index URL above.
* **Out of GPU memory** - lower `--batch-size` (teacher) or run the training scripts one at a time.
* **An interrupted training run** - restart the same command with `--resume`.

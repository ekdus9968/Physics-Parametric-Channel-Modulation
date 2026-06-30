# PPCM: Physics-Parametric Channel Modulation

Physics-based plug-in module for underwater object detection.
No retraining required. Works with any pretrained detector.

---

## Environment

- Python 3.10.11
- PyTorch 2.12.1+cu132
- CUDA 13.2
- GPU: NVIDIA GeForce RTX 5060 Ti

---

## Installation

```bash
python -m venv ppcm_env
ppcm_env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu132
python -m pip install numpy==1.26.4 opencv-python matplotlib tqdm scipy
python -m pip install pycocotools
```

---

## Dataset

S-UODAC2020 (DMCL benchmark)
- Source: https://github.com/mousecpn/DMC-Domain-Generalization-for-Underwater-Object-Detection
- Train: type1~6 (4,669 images) → instances_source.json
- Test:  type7   (785 images)   → instances_target.json
- Categories: echinus, starfish, holothurian, scallop
data/S-UODAC2020/

├── type1/          # train images

├── type2/

├── type3/

├── type4/

├── type5/

├── type6/

├── type7/          # test images

└── COCO_Annotations/

├── instances_source.json   # train annotations

└── instances_target.json   # test annotations

---

## Project Structure
PPCM/

├── data/

│   └── S-UODAC2020/

├── scripts/

│   ├── check_install.py        # verify environment

│   ├── check_data.py           # verify dataset structure

│   ├── check_filenames.py      # verify annotation filenames

│   ├── check_distribution.py   # verify image distribution across types

│   ├── train_baseline.py       # Phase 1: DeepAll baseline training

│   ├── evaluate_baseline.py    # Phase 1: per-type evaluation

│   └── (to be added)

├── work_dirs/

│   └── baseline/               # saved checkpoints

├── ppcm_env/                   # virtual environment

└── README.md

---

## Pipeline Overview
Input Image

├── Water Type Estimator  → water type t

└── Depth Estimator       → depth map z(x)
Pretrained Detector (frozen)

Conv1

└── PPCM Stage 1: channel weighting (uses t, no depth)

Backbone

FPN → P3, P4

└── PPCM Stage 2: spatial weighting (uses t + z(x))

Detection Head

Prediction

---

## Phases

### Phase 1 — DeepAll Baseline
Goal: Reproduce DeepAll result on S-UODAC2020 (target: mAP@50 ~48.86%)

```bash
# Step 1: verify environment
python scripts/check_install.py

# Step 2: verify dataset
python scripts/check_data.py
python scripts/check_distribution.py

# Step 3: train
python scripts/train_baseline.py

# Step 4: evaluate
python scripts/evaluate_baseline.py
```

result: Type 7 Evaluation - 46.92%

### Phase 2 — Water Type & Depth Estimator Validation
Goal: Verify reliability of PPCM inputs (coming soon)

### Phase 3 — PPCM Stage 1
Goal: Channel weighting via W1 bridge (coming soon)

### Phase 4 — PPCM Stage 2
Goal: Spatial weighting via depth map (coming soon)

### Phase 5 — Full Evaluation & Ablation
Goal: Compare PPCM against baselines on S-UODAC2020 (coming soon)

---

## Expected Results (Phase 1 Target)

| Method   | Backbone   | mAP@50 |
|----------|------------|--------|
| DeepAll  | ResNet-50  | 48.86% |
| DANN     | ResNet-50  | 37.32% |
| DG-YOLO  | DarkNet-53 | 39.24% |
| DMCL     | ResNet-50  | 61.36% |
| DSP      | ResNet-50  | 61.88% |
| PPCM     | ResNet-50  | TBD    |
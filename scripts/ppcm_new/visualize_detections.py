"""
Visualize detection results on type7 test images.
Compares all available models: Baseline vs Stage 1 vs Stage 2 vs Full PPCM

Usage:
    python scripts/ppcm_new/visualize_detections.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import random

from model   import PPCMPipeline
from dataset import estimate_water_type, estimate_depth

# ── Config ────────────────────────────────────────────────────────────
BASE_PATH    = 'data/S-UODAC2020'
WORK_DIR     = 'work_dirs/ppcm_new'
VIZ_DIR      = os.path.join(WORK_DIR, 'viz', 'detection_compare')
NUM_IMAGES   = 6
SCORE_THRESH = 0.4
NUM_CLASSES  = 5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(VIZ_DIR, exist_ok=True)

CATEGORIES = {1: 'echinus', 2: 'starfish', 3: 'holothurian', 4: 'scallop'}
COLORS     = {1: 'red', 2: 'cyan', 3: 'yellow', 4: 'magenta'}

# ── Model configs — load whatever checkpoints exist ───────────────────
MODEL_CONFIGS = [
    ('s1=False_s2=False', False, False, 'Baseline'),
    ('s1=True_s2=False',  True,  False, 'Stage 1'),
    ('s1=False_s2=True',  False, True,  'Stage 2'),
    ('s1=True_s2=True',   True,  True,  'Full PPCM'),
]

def load_model(checkpoint_path, use_stage1, use_stage2):
    model = PPCMPipeline(
        num_classes = NUM_CLASSES,
        alpha       = 0.5,
        depth_scale = 1.0,
        use_stage1  = use_stage1,
        use_stage2  = use_stage2
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model

print("Loading available models...")
models = {}
for mode, s1, s2, label in MODEL_CONFIGS:
    ckpt = os.path.join(WORK_DIR, f'{mode}_epoch_12.pth')
    if os.path.exists(ckpt):
        models[label] = (load_model(ckpt, s1, s2), s1, s2)
        print(f"  ✓ {label}")
    else:
        print(f"  ✗ {label} — not found")

if not models:
    print("No checkpoints found.")
    exit()

# ── Load test images ──────────────────────────────────────────────────
ann_file        = os.path.join(BASE_PATH, 'COCO_Annotations', 'instances_target.json')
coco_gt         = COCO(ann_file)
img_ids_with_ann = list(set([ann['image_id'] for ann in coco_gt.anns.values()]))
random.seed(42)
selected_ids = random.sample(img_ids_with_ann, min(NUM_IMAGES, len(img_ids_with_ann)))

# ── Draw boxes helper ─────────────────────────────────────────────────
def draw_boxes(ax, output, score_thresh, title, is_gt=False, anns=None):
    if is_gt:
        count = 0
        for ann in anns:
            x, y, w, h = ann['bbox']
            color = COLORS.get(ann['category_id'], 'white')
            ax.add_patch(patches.Rectangle(
                (x,y), w, h, linewidth=2, edgecolor=color, facecolor='none'
            ))
            cat = CATEGORIES.get(ann['category_id'], str(ann['category_id']))
            ax.text(x, max(y-4,0), cat, color=color, fontsize=7,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
            count += 1
        ax.set_title(f'{title}\n({count} objects)', fontsize=9)
    else:
        boxes  = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        count  = 0
        for box, score, label in zip(boxes, scores, labels):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box
            color = COLORS.get(int(label), 'white')
            ax.add_patch(patches.Rectangle(
                (x1,y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            ))
            cat = CATEGORIES.get(int(label), str(label))
            ax.text(x1, max(y1-4,0), f'{cat} {score:.2f}',
                    color=color, fontsize=7,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
            count += 1
        ax.set_title(f'{title}\n({count} det >{score_thresh:.0%})', fontsize=9)

    ax.axis('off')
    return count

# ── Main loop ─────────────────────────────────────────────────────────
n_cols = 1 + len(models)  # GT + each model

print(f"\nRunning inference on {len(selected_ids)} images...")

for img_id in selected_ids:
    img_info = coco_gt.imgs[img_id]
    fname    = img_info['file_name']
    img_path = os.path.join(BASE_PATH, 'type7', fname)

    if not os.path.exists(img_path):
        print(f"  Skipping {fname}")
        continue

    # Load
    img    = cv2.imread(img_path)
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = img.astype(np.float32) / 255.0
    img_t  = torch.from_numpy(img_np).permute(2,0,1)

    water_type = estimate_water_type(img_path)
    depth_map  = estimate_depth(img_path, str(DEVICE))

    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns    = coco_gt.loadAnns(ann_ids)

    # Run all models
    outputs = {}
    for label, (model, s1, s2) in models.items():
        with torch.no_grad():
            out = model(
                images      = [img_t.to(DEVICE)],
                depth_maps  = [depth_map.to(DEVICE)],
                water_types = [water_type]
            )
        outputs[label] = out[0]

    # Plot
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))

    # GT
    axes[0].imshow(img_np)
    draw_boxes(axes[0], None, SCORE_THRESH, 'Ground Truth',
               is_gt=True, anns=anns)

    # Each model
    counts = {}
    for ax, (label, output) in zip(axes[1:], outputs.items()):
        ax.imshow(img_np)
        c = draw_boxes(ax, output, SCORE_THRESH, label)
        counts[label] = c

    count_str = '  '.join([f'{k}={v}' for k,v in counts.items()])
    plt.suptitle(
        f'{fname} | wt={water_type} | GT={len(anns)}\n{count_str}',
        fontsize=10
    )
    plt.tight_layout()

    save_path = os.path.join(VIZ_DIR, f'{img_id:06d}_{fname}')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

print(f"\nSaved to: {VIZ_DIR}")
print("\nColor: red=echinus  cyan=starfish  yellow=holothurian  magenta=scallop")
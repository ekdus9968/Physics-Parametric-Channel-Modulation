import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter

from scripts.ppcm_old.water_type_estimator import estimate_water_type

BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/baseline'
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

class UnderwaterDataset(Dataset):
    def __init__(self, ann_file, base_path, type_dirs):
        self.coco      = COCO(ann_file)
        self.base_path = base_path
        self.type_dirs = type_dirs
        self.img_ids   = list(self.coco.imgs.keys())

    def find_image(self, fname):
        for t in self.type_dirs:
            path = os.path.join(self.base_path, t, fname)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.find_image(img_info['file_name'])
        img      = cv2.imread(img_path)
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img      = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2,0,1), img_id, img_path

def collate_fn(batch):
    return tuple(zip(*batch))

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    ckpt  = torch.load(os.path.join(WORK_DIR, 'epoch_12.pth'),
                       map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model

def compute_channel_weights(S_normalized, water_type, alpha=0.1):
    beta = IOP_TABLE[water_type]
    rel  = torch.tensor([
        1.0/beta['R'], 1.0/beta['G'], 1.0/beta['B']
    ], device=S_normalized.device)
    rel     = rel / rel.sum()
    weights = S_normalized @ rel
    weights = weights / weights.mean()
    weights = 1.0 + alpha * (weights - 1.0)
    return weights

def evaluate_with_config(water_type_override=None,
                         alpha=0.1,
                         use_ppcm=True,
                         label=''):
    model = load_model()
    conv1_w = model.backbone.body.conv1.weight.data
    w1 = torch.norm(conv1_w.detach().view(64,3,-1), dim=2)
    S  = F.softmax(w1, dim=1)
    current_wt = ['III']

    if use_ppcm:
        def hook_fn(module, input, output):
            wt = water_type_override if water_type_override \
                 else current_wt[0]
            weights = compute_channel_weights(S, wt, alpha)
            return output * weights.view(1,64,1,1)
        hook = model.backbone.body.conv1.register_forward_hook(hook_fn)

    dataset = UnderwaterDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations',
                                 'instances_target.json'),
        base_path = BASE_PATH,
        type_dirs = ['type7']
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    results = []
    wt_counter = Counter()

    with torch.no_grad():
        for imgs, img_ids, img_paths in tqdm(loader, desc=label):
            imgs = [img.to(DEVICE) for img in imgs]

            if use_ppcm and not water_type_override:
                wt, _ = estimate_water_type(img_paths[0])
                current_wt[0] = wt
                wt_counter[wt] += 1

            outputs = model(imgs)
            for img_id, output in zip(img_ids, outputs):
                boxes  = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, lbl in zip(boxes, scores, labels):
                    x1,y1,x2,y2 = box
                    results.append({
                        'image_id':    int(img_id),
                        'category_id': int(lbl),
                        'bbox': [float(x1),float(y1),
                                 float(x2-x1),float(y2-y1)],
                        'score': float(score)
                    })

    if use_ppcm and not water_type_override:
        print(f"  Water type distribution: {dict(wt_counter)}")

    if use_ppcm:
        hook.remove()

    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[1] * 100  # mAP@50

if __name__ == '__main__':
    print("=" * 60)
    print("PPCM Diagnosis: Finding What Hurts Performance")
    print("=" * 60)

    results = {}

    # 1. Baseline
    print("\n[1] Baseline (no PPCM)")
    results['baseline'] = evaluate_with_config(
        use_ppcm=False, label='Baseline'
    )

    # 2. PPCM with estimated water type
    print("\n[2] PPCM - estimated water type (current method)")
    results['ppcm_estimated'] = evaluate_with_config(
        use_ppcm=True, alpha=0.1,
        label='PPCM estimated wt'
    )

    # 3. PPCM with each fixed water type
    # Tests if wrong water type assignment is the problem
    print("\n[3] PPCM - fixed water type sweep")
    for wt in ['I', 'II', 'III', '1C', '5C', '9C']:
        key = f'ppcm_fixed_{wt}'
        print(f"\n  Fixed type={wt}")
        results[key] = evaluate_with_config(
            use_ppcm=True, alpha=0.1,
            water_type_override=wt,
            label=f'Fixed {wt}'
        )

    # 4. Alpha sweep on best water type
    print("\n[4] Alpha sweep (fixed type=I, as type7→I)")
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
        key = f'alpha_{alpha}_typeI'
        results[key] = evaluate_with_config(
            use_ppcm=True, alpha=alpha,
            water_type_override='I',
            label=f'alpha={alpha} type=I'
        )

    # 5. Uniform weighting (all channels equal)
    # Tests if non-uniform weighting is the problem
    print("\n[5] Uniform weighting (alpha=0, PPCM on but no effect)")
    results['ppcm_uniform'] = evaluate_with_config(
        use_ppcm=True, alpha=0.0,
        water_type_override='III',
        label='Uniform (alpha=0)'
    )

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Diagnosis Summary")
    print("=" * 60)
    baseline = results['baseline']
    for key, val in results.items():
        delta  = val - baseline
        marker = ' ← best' if val == max(results.values()) else ''
        print(f"{key:<35} {val:.2f}%  ({delta:+.2f}%){marker}")
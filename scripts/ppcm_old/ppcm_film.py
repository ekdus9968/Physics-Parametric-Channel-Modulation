"""
This tests FiLM applied after BN. If this also gives no improvement, 
then we know the problem is not BN absorption 
but something more fundamental about where in the network physics information needs to go.
"""

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
from scripts.ppcm_old.water_type_estimator import estimate_water_type

BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/baseline'
OUTPUT_DIR  = 'work_dirs/ppcm_film'
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

def compute_film_params(water_type, num_channels=64):
    """
    Compute physics-derived FiLM parameters.
    Applied AFTER BN: output = gamma * BN(x) + beta

    gamma: scale based on channel reliability
           reliable channels (low attenuation) get gamma > 1
           unreliable channels (high attenuation) get gamma < 1

    beta:  shift based on backscatter B_inf
           channels with high backscatter get positive beta
           (compensates for additive backscatter component)
    """
    beta_d = IOP_TABLE[water_type]

    # Channel reliability: exp(-beta_D) normalized
    # Higher reliability = gamma > 1 (amplify)
    # Lower reliability  = gamma < 1 (suppress)
    rel = torch.tensor([
        1.0 / beta_d['R'],
        1.0 / beta_d['G'],
        1.0 / beta_d['B']
    ])
    rel = rel / rel.mean()  # normalize around 1.0

    # gamma: uniform across feature channels
    # All 64 channels get same gamma vector (3 values mapped to 64)
    # Simple: use mean reliability as global scale per RGB component
    # For now: single gamma scalar per water type
    gamma_scalar = rel.mean().item()  # scalar > 1 = reliable, < 1 = not

    # beta: backscatter compensation
    # B_inf values represent additive veiling light
    # We subtract a small proportion to compensate
    b_inf_mean = {
        'I':   0.117, 'II':  0.123, 'III': 0.133,
        '1C':  0.137, '5C':  0.147, '9C':  0.163
    }
    beta_scalar = -b_inf_mean[water_type] * 0.1  # small negative shift

    return gamma_scalar, beta_scalar


class PPCMFiLM(nn.Module):
    """
    Physics-parametric FiLM modulation.
    Applied AFTER BN — BN cannot absorb this.

    output = gamma(t) * BN(x) + beta(t)

    gamma(t): physics-derived channel reliability scale
    beta(t):  physics-derived backscatter compensation shift
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha  # strength control

    def forward(self, feature_map, water_type):
        """
        feature_map: (B, C, H, W) — output of BN
        water_type:  Jerlov type string
        """
        gamma, beta = compute_film_params(water_type)

        # Interpolate toward identity with alpha
        # alpha=0: no change
        # alpha=1: full physics modulation
        effective_gamma = 1.0 + self.alpha * (gamma - 1.0)
        effective_beta  = self.alpha * beta

        return feature_map * effective_gamma + effective_beta


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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2,0,1), img_id, img_path

def collate_fn(batch):
    return tuple(zip(*batch))

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    ckpt  = torch.load(
        os.path.join(WORK_DIR, 'epoch_12.pth'),
        map_location=DEVICE
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model

def find_bn_after_conv1(model):
    """
    Find the BN layer immediately after Conv1 in ResNet50.
    ResNet: conv1 → bn1 → relu → maxpool
    Hook is placed on bn1 output.
    """
    return model.backbone.body.bn1

def evaluate_film(alpha=1.0,
                  water_type_override=None,
                  label=''):
    model   = load_model()
    film    = PPCMFiLM(alpha=alpha).to(DEVICE)
    bn1     = find_bn_after_conv1(model)
    current_wt = ['III']

    def hook_fn(module, input, output):
        wt = water_type_override if water_type_override \
             else current_wt[0]
        return film(output, wt)

    hook = bn1.register_forward_hook(hook_fn)

    dataset = UnderwaterDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations',
                                 'instances_target.json'),
        base_path = BASE_PATH,
        type_dirs = ['type7']
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    results = []
    with torch.no_grad():
        for imgs, img_ids, img_paths in tqdm(loader, desc=label):
            imgs = [img.to(DEVICE) for img in imgs]

            if not water_type_override:
                wt, _ = estimate_water_type(img_paths[0])
                current_wt[0] = wt

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

    hook.remove()

    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[1] * 100

if __name__ == '__main__':
    print("=" * 60)
    print("PPCM-FiLM: Post-BN Physics Modulation")
    print("=" * 60)

    results = {}

    # Baseline
    print("\n[0] Baseline")
    model = load_model()
    dataset = UnderwaterDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations',
                                 'instances_target.json'),
        base_path = BASE_PATH,
        type_dirs = ['type7']
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    res = []
    with torch.no_grad():
        for imgs, img_ids, img_paths in tqdm(loader, desc='Baseline'):
            imgs    = [img.to(DEVICE) for img in imgs]
            outputs = model(imgs)
            for img_id, output in zip(img_ids, outputs):
                for box, score, lbl in zip(
                        output['boxes'].cpu().numpy(),
                        output['scores'].cpu().numpy(),
                        output['labels'].cpu().numpy()):
                    x1,y1,x2,y2 = box
                    res.append({
                        'image_id': int(img_id),
                        'category_id': int(lbl),
                        'bbox': [float(x1),float(y1),
                                 float(x2-x1),float(y2-y1)],
                        'score': float(score)
                    })
    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(res)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results['baseline'] = coco_eval.stats[1] * 100

    # FiLM alpha sweep with estimated water type
    print("\n[1] FiLM - alpha sweep (estimated water type)")
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]:
        key = f'film_alpha{alpha}_estimated'
        results[key] = evaluate_film(
            alpha=alpha,
            label=f'FiLM alpha={alpha}'
        )

    # FiLM fixed water type sweep
    print("\n[2] FiLM - fixed water type (alpha=1.0)")
    for wt in ['I','II','III','1C','5C','9C']:
        key = f'film_fixed_{wt}'
        results[key] = evaluate_film(
            alpha=1.0,
            water_type_override=wt,
            label=f'FiLM fixed={wt}'
        )

    # Summary
    print("\n" + "=" * 60)
    print("FiLM Results Summary")
    print("=" * 60)
    baseline = results['baseline']
    best_key = max(results, key=results.get)
    for key, val in results.items():
        delta  = val - baseline
        marker = ' ← best' if key == best_key else ''
        print(f"{key:<40} {val:.2f}%  ({delta:+.2f}%){marker}")
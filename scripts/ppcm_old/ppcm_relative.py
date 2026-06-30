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

# Training distribution — type1~6 mostly classified as 5C
# This is what the model's features are calibrated to
TRAIN_WATER_TYPE = '5C'

def compute_relative_weights(S_normalized,
                              train_wt, test_wt,
                              alpha=0.1):
    """
    Compute per-feature-channel correction weights.

    Direction: make test features look like train features.
    reliability_train / reliability_test per RGB channel,
    then map to feature channels via W1 bridge.

    If reliability_train < reliability_test:
        → suppress this channel (test has more signal than model expects)
    If reliability_train > reliability_test:
        → boost this channel (test has less signal than model expects)
    """
    beta_train = IOP_TABLE[train_wt]
    beta_test  = IOP_TABLE[test_wt]

    # Relative reliability: train / test
    # = exp(-beta_train) / exp(-beta_test)
    # = exp(beta_test - beta_train)
    rel_R = (beta_test['R'] - beta_train['R'])  # positive = boost R
    rel_G = (beta_test['G'] - beta_train['G'])
    rel_B = (beta_test['B'] - beta_train['B'])

    # Convert to multiplicative weights centered at 1.0
    # exp(delta_beta) gives the relative attenuation difference
    rel = torch.tensor([
        np.exp(rel_R),
        np.exp(rel_G),
        np.exp(rel_B)
    ], device=S_normalized.device, dtype=torch.float32)

    # Normalize around 1.0
    rel = rel / rel.mean()

    # Map to 64 feature channels via W1 bridge
    weights = S_normalized @ rel  # (64,)
    weights = weights / weights.mean()

    # Alpha scaling
    weights = 1.0 + alpha * (weights - 1.0)
    return weights


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

def evaluate_relative(alpha=0.1,
                      test_wt_override=None,
                      train_wt=TRAIN_WATER_TYPE,
                      apply_after_bn=False,
                      label=''):
    model = load_model()

    # Extract W1 sensitivity
    w1  = model.backbone.body.conv1.weight.data
    S   = torch.norm(w1.detach().view(64,3,-1), dim=2)
    S_n = F.softmax(S, dim=1).to(DEVICE)

    current_wt = ['I']  # type7 → I

    if apply_after_bn:
        # Hook on bn1 output (post-BN)
        target_module = model.backbone.body.bn1
    else:
        # Hook on conv1 output (pre-BN)
        target_module = model.backbone.body.conv1

    def hook_fn(module, input, output):
        test_wt = test_wt_override if test_wt_override \
                  else current_wt[0]
        weights = compute_relative_weights(
            S_n, train_wt, test_wt, alpha
        )
        return output * weights.view(1, 64, 1, 1)

    hook = target_module.register_forward_hook(hook_fn)

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
            if not test_wt_override:
                wt, _ = estimate_water_type(img_paths[0])
                current_wt[0] = wt
            outputs = model(imgs)
            for img_id, output in zip(img_ids, outputs):
                for box, score, lbl in zip(
                        output['boxes'].cpu().numpy(),
                        output['scores'].cpu().numpy(),
                        output['labels'].cpu().numpy()):
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
    print("PPCM Relative Correction: test → train distribution")
    print(f"Train reference type: {TRAIN_WATER_TYPE}")
    print("=" * 60)

    results = {}

    # Baseline
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
        for imgs, img_ids, _ in tqdm(loader, desc='Baseline'):
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
    print(f"Baseline: {results['baseline']:.2f}%")

    # Alpha sweep — pre-BN, relative correction, fixed I→5C
    print("\n[1] Pre-BN relative correction (type I → train 5C)")
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        key = f'prebn_rel_alpha{alpha}'
        results[key] = evaluate_relative(
            alpha=alpha,
            test_wt_override='I',
            train_wt='5C',
            apply_after_bn=False,
            label=f'pre-BN alpha={alpha}'
        )

    # Alpha sweep — post-BN, relative correction, fixed I→5C
    print("\n[2] Post-BN relative correction (type I → train 5C)")
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        key = f'postbn_rel_alpha{alpha}'
        results[key] = evaluate_relative(
            alpha=alpha,
            test_wt_override='I',
            train_wt='5C',
            apply_after_bn=True,
            label=f'post-BN alpha={alpha}'
        )

    # Try different train reference types
    print("\n[3] Different train reference types (alpha=0.1, post-BN)")
    for train_wt in ['III', '5C', '9C', '1C']:
        key = f'postbn_train{train_wt}'
        results[key] = evaluate_relative(
            alpha=0.1,
            test_wt_override='I',
            train_wt=train_wt,
            apply_after_bn=True,
            label=f'train_ref={train_wt}'
        )

    print("\n" + "=" * 60)
    print("Relative Correction Summary")
    print("=" * 60)
    baseline = results['baseline']
    best_key = max(results, key=results.get)
    for key, val in results.items():
        delta  = val - baseline
        marker = ' ← best' if key == best_key else ''
        print(f"{key:<40} {val:.2f}%  ({delta:+.2f}%){marker}")
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

# Config
BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/baseline'
OUTPUT_DIR  = 'work_dirs/ppcm_alpha'
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

class PPCMStage1Alpha(nn.Module):
    """PPCM Stage 1 with alpha scaling for weighting strength."""
    def __init__(self, conv1_weight, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        w1 = conv1_weight.detach()
        S  = torch.norm(w1.view(64, 3, -1), dim=2)
        S_normalized = F.softmax(S, dim=1)
        self.register_buffer('S_normalized', S_normalized)

    def compute_weights(self, water_type):
        beta = IOP_TABLE[water_type]
        reliability = torch.tensor([
            1.0 / beta['R'],
            1.0 / beta['G'],
            1.0 / beta['B']
        ], device=self.S_normalized.device)
        reliability = reliability / reliability.sum()
        weights = self.S_normalized @ reliability   # (64,)
        weights = weights / weights.mean()          # normalize to mean=1

        # Alpha scaling: 0=no change, 1=full weighting
        weights = 1.0 + self.alpha * (weights - 1.0)
        return weights

    def forward(self, feature_map, water_type):
        weights = self.compute_weights(water_type)
        weights = weights.view(1, 64, 1, 1)
        return feature_map * weights

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

def load_base_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    checkpoint = torch.load(
        os.path.join(WORK_DIR, 'epoch_12.pth'),
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model

def evaluate_alpha(alpha):
    model    = load_base_model()
    conv1_w  = model.backbone.body.conv1.weight.data
    ppcm_s1  = PPCMStage1Alpha(conv1_w, alpha=alpha).to(DEVICE)
    current_wt = ['III']

    def hook_fn(module, input, output):
        return ppcm_s1(output, current_wt[0])

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
    with torch.no_grad():
        for imgs, img_ids, img_paths in tqdm(
                loader, desc=f'alpha={alpha:.1f}'):
            imgs = [img.to(DEVICE) for img in imgs]
            wt, _ = estimate_water_type(img_paths[0])
            current_wt[0] = wt
            outputs = model(imgs)

            for img_id, output in zip(img_ids, outputs):
                boxes  = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, lbl in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        'image_id':    int(img_id),
                        'category_id': int(lbl),
                        'bbox': [float(x1), float(y1),
                                 float(x2-x1), float(y2-y1)],
                        'score': float(score)
                    })

    hook.remove()

    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[1]  # mAP@50

if __name__ == '__main__':
    print("=" * 50)
    print("Option A: Alpha Scaling Test")
    print("=" * 50)

    # alpha=0.0 is identical to baseline
    alphas  = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for alpha in alphas:
        print(f"\n--- alpha = {alpha} ---")
        mAP = evaluate_alpha(alpha)
        results[alpha] = mAP * 100
        print(f"mAP@50: {mAP*100:.2f}%")

    print("\n" + "=" * 50)
    print("Alpha Scaling Summary")
    print("=" * 50)
    print(f"{'Alpha':<10} {'mAP@50':>10} {'Delta':>10}")
    print("-" * 30)
    baseline = results[0.0]
    for alpha, mAP in results.items():
        delta = mAP - baseline
        marker = ' ← best' if mAP == max(results.values()) else ''
        print(f"{alpha:<10} {mAP:>10.2f}% {delta:>+10.2f}%{marker}")

    import json
    with open(os.path.join(OUTPUT_DIR, 'alpha_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/alpha_results.json")
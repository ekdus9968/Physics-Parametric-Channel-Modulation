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

BASE_PATH   = 'data/RUOD'
NUM_CLASSES = 11
ALPHA       = 0.1
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

class PPCMStage1(nn.Module):
    def __init__(self, conv1_weight, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        w1 = conv1_weight.detach()
        S  = torch.norm(w1.view(64, 3, -1), dim=2)
        self.register_buffer('S_normalized', F.softmax(S, dim=1))

    def compute_weights(self, water_type):
        beta = IOP_TABLE[water_type]
        reliability = torch.tensor([
            1.0 / beta['R'],
            1.0 / beta['G'],
            1.0 / beta['B']
        ], device=self.S_normalized.device)
        reliability = reliability / reliability.sum()
        weights = self.S_normalized @ reliability
        weights = weights / weights.mean()
        weights = 1.0 + self.alpha * (weights - 1.0)
        return weights

    def forward(self, feature_map, water_type):
        weights = self.compute_weights(water_type)
        return feature_map * weights.view(1, 64, 1, 1)

class RUODDataset(Dataset):
    def __init__(self, ann_file, img_dir):
        self.coco    = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2,0,1), img_id, img_path

def collate_fn(batch):
    return tuple(zip(*batch))

def load_model(checkpoint_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model

def evaluate(checkpoint_path, use_ppcm=False, label=''):
    model = load_model(checkpoint_path)

    if use_ppcm:
        conv1_w    = model.backbone.body.conv1.weight.data
        ppcm_s1    = PPCMStage1(conv1_w, alpha=ALPHA).to(DEVICE)
        current_wt = ['III']

        def hook_fn(module, input, output):
            return ppcm_s1(output, current_wt[0])

        hook = model.backbone.body.conv1.register_forward_hook(hook_fn)

    dataset = RUODDataset(
        ann_file = os.path.join(BASE_PATH, 'RUOD_ANN', 'instances_test.json'),
        img_dir  = os.path.join(BASE_PATH, 'RUOD_pic', 'test')
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    results = []
    with torch.no_grad():
        for imgs, img_ids, img_paths in tqdm(loader, desc=label):
            imgs = [img.to(DEVICE) for img in imgs]

            if use_ppcm:
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

    if use_ppcm:
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
    print("Option C: RUOD Evaluation")
    print("=" * 50)

    ckpt = 'work_dirs/ruod_baseline/epoch_12.pth'

    results = {}

    print("\n[1] Baseline, no PPCM")
    results['baseline'] = evaluate(ckpt, use_ppcm=False,
                                   label='Baseline') * 100

    print("\n[2] Baseline + PPCM Stage 1")
    results['baseline+ppcm'] = evaluate(ckpt, use_ppcm=True,
                                        label='PPCM Stage 1') * 100

    print("\n" + "=" * 50)
    print("RUOD Results Summary")
    print("=" * 50)
    baseline = results['baseline']
    for key, val in results.items():
        delta  = val - baseline
        marker = ' ← best' if val == max(results.values()) else ''
        print(f"{key:<25} {val:.2f}%  ({delta:+.2f}%){marker}")
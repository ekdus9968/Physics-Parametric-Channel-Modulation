import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import pipeline as hf_pipeline
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm

from scripts.ppcm_old.water_type_estimator import estimate_water_type
from scripts.ppcm_old.train_ppcm_yolo import (
    PPCMYOLOWrapper, PPCMAnalytic, CorrectionHeader,
    get_dominant_beta, IOP_TABLE, DEPTH_MIN, DEPTH_MAX
)

# Config
YOLO_WEIGHTS = 'runs/detect/work_dirs/yolo_ruod_baseline-3/weights/best.pt'
PPCM_WEIGHTS = 'work_dirs/ppcm_yolo/best.pth'
BASE_PATH    = 'data/RUOD'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# Load depth estimator
print("Loading depth estimator...")
depth_pipe = hf_pipeline(
    task   = "depth-estimation",
    model  = "depth-anything/Depth-Anything-V2-Small-hf",
    device = 0 if str(DEVICE) == 'cuda' else -1
)
print("Done.")

def get_depth(img_path):
    pil  = Image.open(img_path).convert('RGB')
    res  = depth_pipe(pil)
    d    = np.array(res['depth']).astype(np.float32)
    dmin, dmax = d.min(), d.max()
    if dmax > dmin:
        d = (d - dmin) / (dmax - dmin)
    return torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(DEVICE)

class RUODTestDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_size=640):
        self.coco     = COCO(ann_file)
        self.img_dir  = img_dir
        self.img_ids  = list(self.coco.imgs.keys())
        self.img_size = img_size

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        W0, H0   = img_info['width'], img_info['height']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        return (torch.from_numpy(img).permute(2,0,1),
                img_id, img_path, W0, H0)

def collate_fn(batch):
    return tuple(zip(*batch))

def run_inference(model, imgs, depth, water_type, use_ppcm):
    """Run forward pass with or without PPCM."""
    imgs_batch = torch.stack(imgs).to(DEVICE)

    with torch.no_grad():
        feats = model._forward_backbone_neck(imgs_batch)

        if use_ppcm:
            d = F.interpolate(depth, size=(640,640),
                              mode='bilinear', align_corners=False)
            feats = model.ppcm(feats, d, water_type)
            feats = model.header(feats)

        out = model.yolo_layers[22](feats)

    return out

def postprocess(out, conf=0.25, iou=0.45):
    """Simple NMS postprocessing for YOLOv8 output."""
    from torchvision.ops import nms

    results = []
    # out shape: (batch, num_classes+4, anchors)
    preds = out[0] if isinstance(out, (list, tuple)) else out

    if preds.dim() == 3:
        preds = preds.permute(0, 2, 1)  # (B, anchors, 4+nc)

    for b in range(preds.shape[0]):
        pred = preds[b]             # (anchors, 4+nc)
        box  = pred[:, :4]          # cx,cy,w,h
        cls  = pred[:, 4:]          # class scores

        scores, labels = cls.max(dim=1)
        mask   = scores > conf
        box    = box[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(box) == 0:
            results.append(([], [], []))
            continue

        # Convert cx,cy,w,h to x1,y1,x2,y2
        x1 = box[:,0] - box[:,2]/2
        y1 = box[:,1] - box[:,3]/2
        x2 = box[:,0] + box[:,2]/2
        y2 = box[:,1] + box[:,3]/2
        boxes_xyxy = torch.stack([x1,y1,x2,y2], dim=1)

        keep = nms(boxes_xyxy, scores, iou)
        results.append((
            boxes_xyxy[keep].cpu().numpy(),
            scores[keep].cpu().numpy(),
            labels[keep].cpu().numpy()
        ))

    return results

def evaluate(use_ppcm=False, label='baseline'):
    # Load model
    yolo_base = YOLO(YOLO_WEIGHTS)
    yolo_base.model.to(DEVICE)
    yolo_base.model.eval()

    model = PPCMYOLOWrapper(yolo_base).to(DEVICE)

    if use_ppcm:
        ckpt = torch.load(PPCM_WEIGHTS, map_location=DEVICE)
        model.header.load_state_dict(ckpt['header_state'])
        print(f"Loaded header from epoch {ckpt['epoch']}, "
              f"loss={ckpt['loss']:.6f}")
    model.eval()

    # Dataset
    dataset = RUODTestDataset(
        ann_file = os.path.join(BASE_PATH, 'RUOD_ANN',
                                'instances_test.json'),
        img_dir  = os.path.join(BASE_PATH, 'RUOD_pic', 'test')
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    results = []
    coco_gt = dataset.coco

    with torch.no_grad():
        for imgs, img_ids, img_paths, W0s, H0s in tqdm(
                loader, desc=f'[{label}]'):

            img_id   = img_ids[0]
            img_path = img_paths[0]
            W0, H0   = W0s[0], H0s[0]

            wt = 'III'
            if use_ppcm:
                wt, _ = estimate_water_type(img_path)
                depth  = get_depth(img_path)
            else:
                depth = None

            out = run_inference(model, imgs, depth, wt, use_ppcm)
            preds = postprocess(out)

            for (boxes, scores, labels) in preds:
                for box, score, lbl in zip(boxes, scores, labels):
                    # Scale back to original image size
                    x1 = float(box[0]) / 640 * W0
                    y1 = float(box[1]) / 640 * H0
                    x2 = float(box[2]) / 640 * W0
                    y2 = float(box[3]) / 640 * H0
                    results.append({
                        'image_id':    int(img_id),
                        'category_id': int(lbl) + 1,  # 1-indexed
                        'bbox':        [x1, y1, x2-x1, y2-y1],
                        'score':       float(score)
                    })

    if len(results) == 0:
        print("No detections.")
        return 0.0

    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[1]  # mAP@50

if __name__ == '__main__':
    print("=" * 50)
    print("PPCM + YOLOv8 Evaluation")
    print("=" * 50)

    print("\n[1] YOLOv8 baseline (no PPCM)")
    mAP_base = evaluate(use_ppcm=False, label='baseline') * 100

    print("\n[2] YOLOv8 + PPCM + Header")
    mAP_ppcm = evaluate(use_ppcm=True, label='PPCM') * 100

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"YOLOv8 baseline:        {mAP_base:.2f}%")
    print(f"YOLOv8 + PPCM + Header: {mAP_ppcm:.2f}%")
    print(f"Delta:                  {mAP_ppcm-mAP_base:+.2f}%")
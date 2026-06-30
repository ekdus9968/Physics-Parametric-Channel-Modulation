import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ppcm_old.ppcm_stage1 import PPCMStage1
from scripts.ppcm_old.water_type_estimator import estimate_water_type

# Config
BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/baseline'
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor, img_id, img_path

def collate_fn(batch):
    return tuple(zip(*batch))

class FasterRCNNWithPPCMS1(nn.Module):
    """
    Faster R-CNN with PPCM Stage 1 inserted after Conv1.
    Backbone frozen. PPCM has no learnable parameters.
    """
    def __init__(self, base_model, ppcm_s1):
        super().__init__()
        self.model   = base_model
        self.ppcm_s1 = ppcm_s1
        self._hook   = None
        self._water_type = 'III'  # default

    def set_water_type(self, water_type):
        self._water_type = water_type

    def _register_hook(self):
        """Register forward hook on Conv1 output."""
        def hook_fn(module, input, output):
            return self.ppcm_s1(output, self._water_type)

        self._hook = self.model.backbone.body.conv1.register_forward_hook(
            hook_fn
        )

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()

    def forward(self, images, targets=None):
        return self.model(images, targets)

def evaluate_with_ppcm(use_ppcm=True):
    """Evaluate model on type7 with or without PPCM Stage 1."""

    # Load base model
    base_model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    checkpoint = torch.load(
        os.path.join(WORK_DIR, 'epoch_12.pth'),
        map_location=DEVICE
    )
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    base_model.to(DEVICE)

    if use_ppcm:
        # Build PPCM Stage 1
        conv1_weight = base_model.backbone.body.conv1.weight.data
        ppcm_s1 = PPCMStage1(conv1_weight).to(DEVICE)
        detector = FasterRCNNWithPPCMS1(base_model, ppcm_s1)
        detector._register_hook()
        label = 'With PPCM Stage 1'
    else:
        detector = base_model
        label = 'Baseline (no PPCM)'

    # Dataset
    dataset = UnderwaterDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations',
                                 'instances_target.json'),
        base_path = BASE_PATH,
        type_dirs = ['type7']
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0
    )

    results = []
    with torch.no_grad():
        for imgs, img_ids, img_paths in tqdm(loader, desc=label):
            imgs = [img.to(DEVICE) for img in imgs]

            # Estimate water type per image
            if use_ppcm:
                wt, _ = estimate_water_type(img_paths[0])
                detector.set_water_type(wt)

            outputs = detector(imgs)

            for img_id, output in zip(img_ids, outputs):
                boxes  = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, lbl in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        'image_id':    int(img_id),
                        'category_id': int(lbl),
                        'bbox':        [float(x1), float(y1),
                                        float(x2-x1), float(y2-y1)],
                        'score':       float(score)
                    })

    if use_ppcm:
        detector.remove_hook()

    # COCO evaluation
    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_50 = coco_eval.stats[1]
    return mAP_50, label

if __name__ == '__main__':
    print("=" * 50)
    print("Phase 3: PPCM Stage 1 Evaluation")
    print("=" * 50)

    # Baseline
    mAP_base, label_base = evaluate_with_ppcm(use_ppcm=False)
    print(f"\n{label_base}: mAP@50 = {mAP_base*100:.2f}%")

    # With PPCM Stage 1
    mAP_ppcm, label_ppcm = evaluate_with_ppcm(use_ppcm=True)
    print(f"{label_ppcm}: mAP@50 = {mAP_ppcm*100:.2f}%")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Baseline:       {mAP_base*100:.2f}%")
    print(f"PPCM Stage 1:   {mAP_ppcm*100:.2f}%")
    print(f"Delta:          {(mAP_ppcm-mAP_base)*100:+.2f}%")
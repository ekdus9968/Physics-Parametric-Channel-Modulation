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
from transformers import pipeline as hf_pipeline
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm

from scripts.ppcm_old.ppcm_stage1 import PPCMStage1
from scripts.ppcm_old.ppcm_stage2 import PPCMStage2
from scripts.ppcm_old.water_type_estimator import estimate_water_type

# Config
BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/baseline'
OUTPUT_DIR  = 'work_dirs/ppcm_full'
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# Load Depth Anything V2
print("Loading depth estimator...")
depth_estimator = hf_pipeline(
    task    = "depth-estimation",
    model   = "depth-anything/Depth-Anything-V2-Small-hf",
    device  = 0 if str(DEVICE) == 'cuda' else -1
)
print("Depth estimator loaded.")

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

def get_depth_map(img_path, device):
    """Get normalized depth map (1, 1, H, W) from Depth Anything V2."""
    pil_img  = Image.open(img_path).convert('RGB')
    result   = depth_estimator(pil_img)
    depth_np = np.array(result['depth']).astype(np.float32)

    # Normalize to [0, 1]
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        depth_np = (depth_np - d_min) / (d_max - d_min)

    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
    return depth_tensor.to(device)

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

def evaluate(mode='baseline'):
    """
    mode: 'baseline' | 'stage1' | 'stage2' | 'full'
    """
    model = load_base_model()

    # Build PPCM modules
    conv1_weight = model.backbone.body.conv1.weight.data
    ppcm_s1 = PPCMStage1(conv1_weight).to(DEVICE)
    ppcm_s2 = PPCMStage2().to(DEVICE)

    hooks = []

    # Stage 1 hook: Conv1 output
    if mode in ['stage1', 'full']:
        current_wt = ['III']  # mutable container for hook closure

        def hook_s1(module, input, output):
            return ppcm_s1(output, current_wt[0])

        hooks.append(
            model.backbone.body.conv1.register_forward_hook(hook_s1)
        )

    # Stage 2 hook: FPN output
    if mode in ['stage2', 'full']:
        current_depth = [None]
        current_wt_s2 = ['III']

        def hook_s2(module, input, output):
            if current_depth[0] is None:
                return output
            return ppcm_s2(output, current_depth[0], current_wt_s2[0])

        hooks.append(
            model.backbone.fpn.register_forward_hook(hook_s2)
        )

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
        for imgs, img_ids, img_paths in tqdm(loader, desc=f'[{mode}]'):
            imgs     = [img.to(DEVICE) for img in imgs]
            img_path = img_paths[0]

            # Estimate water type
            wt, _ = estimate_water_type(img_path)

            # Update hook state
            if mode in ['stage1', 'full']:
                current_wt[0] = wt
            if mode in ['stage2', 'full']:
                current_wt_s2[0] = wt
                current_depth[0] = get_depth_map(img_path, DEVICE)

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
                        'bbox':        [float(x1), float(y1),
                                        float(x2-x1), float(y2-y1)],
                        'score':       float(score)
                    })

    # Remove hooks
    for h in hooks:
        h.remove()

    # COCO evaluation
    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[1]  # mAP@50

if __name__ == '__main__':
    print("=" * 50)
    print("Phase 4: Full PPCM Evaluation")
    print("=" * 50)

    results = {}
    for mode in ['baseline', 'stage1', 'stage2', 'full']:
        print(f"\n--- Mode: {mode} ---")
        mAP = evaluate(mode=mode)
        results[mode] = mAP * 100
        print(f"mAP@50: {mAP*100:.2f}%")

    print("\n" + "=" * 50)
    print("Ablation Summary")
    print("=" * 50)
    print(f"Baseline:          {results['baseline']:.2f}%")
    print(f"Stage 1 only:      {results['stage1']:.2f}%")
    print(f"Stage 2 only:      {results['stage2']:.2f}%")
    print(f"Full PPCM (S1+S2): {results['full']:.2f}%")
    print(f"\nDelta (Full vs Baseline): "
          f"{results['full']-results['baseline']:+.2f}%")

    # Save results
    import json
    save_path = os.path.join(OUTPUT_DIR, 'ablation_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
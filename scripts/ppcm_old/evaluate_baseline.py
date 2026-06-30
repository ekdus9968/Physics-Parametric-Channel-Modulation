import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2
import numpy as np
import json
from tqdm import tqdm

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
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, img_id

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def evaluate(ann_file, type_dirs, checkpoint_path, label):
    dataset = UnderwaterDataset(
        ann_file  = ann_file,
        base_path = BASE_PATH,
        type_dirs = type_dirs
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0
    )

    # Load model
    model = get_model(NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)

    results = []
    with torch.no_grad():
        for imgs, img_ids in tqdm(loader, desc=f'Evaluating {label}'):
            imgs    = [img.to(DEVICE) for img in imgs]
            outputs = model(imgs)

            for img_id, output in zip(img_ids, outputs):
                boxes  = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label_ in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        'image_id':    int(img_id),
                        'category_id': int(label_),
                        'bbox':        [float(x1), float(y1),
                                        float(x2-x1), float(y2-y1)],
                        'score':       float(score)
                    })

    if len(results) == 0:
        print(f"{label}: No detections")
        return 0.0, 0.0

    coco_gt   = dataset.coco
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_5095 = coco_eval.stats[0]  # mAP@50:95
    mAP_50   = coco_eval.stats[1]  # mAP@50
    return mAP_50, mAP_5095

if __name__ == '__main__':
    checkpoint_path = os.path.join(WORK_DIR, 'epoch_12.pth')

    print("=" * 50)
    print("Phase 1: DeepAll Baseline Evaluation")
    print("=" * 50)

    # Test on type7 (target domain)
    mAP_50, mAP_5095 = evaluate(
        ann_file        = os.path.join(BASE_PATH, 'COCO_Annotations', 'instances_target.json'),
        type_dirs       = ['type7'],
        checkpoint_path = checkpoint_path,
        label           = 'type7 (target)'
    )
    print(f"\n[Target Domain - type7]")
    print(f"mAP@50:    {mAP_50:.4f} ({mAP_50*100:.2f}%)")
    print(f"mAP@50:95: {mAP_5095:.4f} ({mAP_5095*100:.2f}%)")
    print(f"\nBaseline target: ~48.86%")
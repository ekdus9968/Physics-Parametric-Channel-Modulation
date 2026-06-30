import sys
sys.path.insert(0, '.')

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm

# Config
BASE_PATH   = 'data/RUOD'
WORK_DIR    = 'work_dirs/ruod_baseline'
NUM_CLASSES = 11       # 10 categories + background
NUM_EPOCHS  = 12
BATCH_SIZE  = 2
LR          = 0.005
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(WORK_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

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
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x+w, y+h])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([img_id])
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    dataset = RUODDataset(
        ann_file = os.path.join(BASE_PATH, 'RUOD_ANN', 'instances_train.json'),
        img_dir  = os.path.join(BASE_PATH, 'RUOD_pic', 'train')
    )
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0
    )
    print(f"Train samples: {len(dataset)}")

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.1
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

        for imgs, targets in pbar:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(imgs, targets)
            losses    = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        torch.save({
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 avg_loss
        }, os.path.join(WORK_DIR, f'epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    print("=" * 50)
    print("Option C: RUOD Baseline Training")
    print("=" * 50)
    train()
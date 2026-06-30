import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm
from scripts.ppcm_old.water_type_estimator import estimate_water_type

# Config
BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/ppcm_trained'
NUM_CLASSES = 5
NUM_EPOCHS  = 12
BATCH_SIZE  = 2
LR          = 0.005
ALPHA       = 0.1   # best alpha from option A
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(WORK_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Alpha:  {ALPHA}")

IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

class PPCMStage1(nn.Module):
    """PPCM Stage 1 with alpha scaling."""
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

class UnderwaterDataset(Dataset):
    def __init__(self, ann_file, base_path):
        self.coco      = COCO(ann_file)
        self.base_path = base_path
        self.img_ids   = list(self.coco.imgs.keys())
        self.type_dirs = ['type1','type2','type3',
                          'type4','type5','type6','type7']

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
        return img_tensor, target, img_path

def collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    return imgs, targets, paths

def train():
    # Load ImageNet pretrained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    model.to(DEVICE)

    # Build PPCM Stage 1
    conv1_weight = model.backbone.body.conv1.weight.data
    ppcm_s1 = PPCMStage1(conv1_weight, alpha=ALPHA).to(DEVICE)
    current_wt = ['III']

    # Register hook — active during training
    def hook_fn(module, input, output):
        return ppcm_s1(output, current_wt[0])

    hook = model.backbone.body.conv1.register_forward_hook(hook_fn)

    # Dataset
    dataset = UnderwaterDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations',
                                 'instances_source.json'),
        base_path = BASE_PATH
    )
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0
    )
    print(f"Train samples: {len(dataset)}")

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

        for imgs, targets, img_paths in pbar:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]

            # Estimate water type for each image in batch
            # Use first image's water type for simplicity
            wt, _ = estimate_water_type(img_paths[0])
            current_wt[0] = wt

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
            'loss':                 avg_loss,
            'alpha':                ALPHA
        }, os.path.join(WORK_DIR, f'epoch_{epoch+1}.pth'))

    hook.remove()
    print("Training complete.")

if __name__ == '__main__':
    print("=" * 50)
    print("Option B: Train with PPCM Stage 1")
    print("=" * 50)
    train()
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
import json

from scripts.ppcm_old.water_type_estimator import estimate_water_type

# Config
YOLO_WEIGHTS = 'runs/detect/work_dirs/yolo_ruod_baseline-3/weights/best.pt'
BASE_PATH    = 'data/RUOD'
WORK_DIR     = 'work_dirs/ppcm_yolo'
NUM_EPOCHS   = 10
LR           = 1e-4
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(WORK_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

DEPTH_MIN = 0.5
DEPTH_MAX  = 10.0

def get_dominant_beta(water_type):
    beta = IOP_TABLE[water_type]
    return (beta['R'] + beta['G'] + beta['B']) / 3.0


# ── PPCM analytic spatial weighting ────────────────────────────
class PPCMAnalytic(nn.Module):
    """
    Physics-parametric spatial weighting.
    No learnable parameters.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, depth_map, water_type):
        """
        features:  list [P3, P4, P5]
        depth_map: (1, 1, H, W) normalized [0,1]
        """
        beta = get_dominant_beta(water_type)
        z    = DEPTH_MIN + depth_map * (DEPTH_MAX - DEPTH_MIN)

        modulated = []
        for feat in features:
            H, W = feat.shape[2], feat.shape[3]
            z_r  = F.interpolate(z, size=(H, W),
                                 mode='bilinear', align_corners=False)
            w    = torch.exp(-beta * z_r)
            w    = w / w.mean()
            modulated.append(feat * w)

        return modulated


# ── Correction Header ───────────────────────────────────────────
class CorrectionHeader(nn.Module):
    """
    Lightweight 1x1 conv per FPN level.
    Only learnable component.
    P3=192, P4=384, P5=576 (YOLOv8m actual channels)
    """
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(192, 192, 1, bias=False),
                nn.BatchNorm2d(192),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(384, 384, 1, bias=False),
                nn.BatchNorm2d(384),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(576, 576, 1, bias=False),
                nn.BatchNorm2d(576),
                nn.SiLU()
            ),
        ])

    def forward(self, features):
        return [conv(feat)
                for conv, feat in zip(self.convs, features)]

    def forward(self, features):
        return [conv(feat)
                for conv, feat in zip(self.convs, features)]


# ── Full PPCM+YOLO wrapper ──────────────────────────────────────
class PPCMYOLOWrapper(nn.Module):
    """
    Frozen YOLOv8 + PPCM (analytic) + CorrectionHeader (learned).
    Only CorrectionHeader trains.
    """
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_layers = yolo_model.model.model   # nn.Sequential

        # Freeze all YOLOv8 layers
        for p in self.yolo_layers.parameters():
            p.requires_grad = False

        self.ppcm   = PPCMAnalytic()
        self.header = CorrectionHeader()

        # Store intermediate outputs
        self._feat_cache = {}

    def _forward_backbone_neck(self, x):
        """
        Forward pass through YOLOv8 layers 0-21.
        Capture outputs at layers 15, 18, 21 (P3, P4, P5).
        Mimic YOLOv8's internal routing (f= indices).
        """
        outputs = {}
        out = x

        for i, layer in enumerate(self.yolo_layers):
            # Handle concat layers (f= list)
            f = layer.f if hasattr(layer, 'f') else -1

            if isinstance(f, list):
                # Collect inputs from specified layer outputs
                inp = [outputs[j] if j != -1 else out
                       for j in f]
                out = layer(inp)
            else:
                out = layer(out)

            outputs[i] = out

            # Stop before Detect layer
            if i == 21:
                break

        # Return P3, P4, P5
        return [outputs[15], outputs[18], outputs[21]]

    def forward(self, x, depth_map=None, water_type='III'):
        # Step 1: Backbone + Neck (frozen)
        with torch.no_grad():
            features = self._forward_backbone_neck(x)

        # Step 2: PPCM analytic weighting
        if depth_map is not None:
            features = self.ppcm(features, depth_map, water_type)

        # Step 3: Correction Header (learned)
        features = self.header(features)

        # Step 4: Detect Head (frozen)
        with torch.no_grad():
            detect_layer = self.yolo_layers[22]
            out = detect_layer(features)

        return out


# ── Dataset ─────────────────────────────────────────────────────
class RUODDataset(Dataset):
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

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        # Resize to 640x640
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        # Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                # Scale to resized image
                x1 = x / W0 * self.img_size
                y1 = y / H0 * self.img_size
                x2 = (x+w) / W0 * self.img_size
                y2 = (y+h) / H0 * self.img_size
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'] - 1)  # 0-indexed

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return img_tensor, boxes, labels, img_path, img_id

def collate_fn(batch):
    return tuple(zip(*batch))


# ── Depth estimator ─────────────────────────────────────────────
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
    t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return t


# ── Training with detection loss ────────────────────────────────
def train():
    # Load base YOLOv8
    yolo_base = YOLO(YOLO_WEIGHTS)
    yolo_base.model.to(DEVICE)
    yolo_base.model.eval()

    # Build PPCM wrapper
    model = PPCMYOLOWrapper(yolo_base).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    # Dataset — images only, no annotation needed for distillation
    class ImageOnlyDataset(Dataset):
        def __init__(self, img_dir, img_size=640):
            self.img_dir  = img_dir
            self.img_size = img_size
            self.images   = [f for f in os.listdir(img_dir)
                             if f.endswith('.jpg')]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.images[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2,0,1), img_path

    def img_collate(batch):
        return tuple(zip(*batch))

    dataset = ImageOnlyDataset(
        img_dir = os.path.join(BASE_PATH, 'RUOD_pic', 'train')
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        collate_fn=img_collate, num_workers=0
    )
    print(f"Train samples: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        model.header.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.header.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

        for imgs, img_paths in pbar:
            imgs = torch.stack(imgs).to(DEVICE)

            # Step 1: Teacher features (original, no PPCM)
            with torch.no_grad():
                teacher_feats = model._forward_backbone_neck(imgs)

            # Step 2: Estimate water type
            wt, _ = estimate_water_type(img_paths[0])

            # Step 3: Get depth map
            depth = get_depth(img_paths[0])
            depth = F.interpolate(
                depth, size=(640, 640),
                mode='bilinear', align_corners=False
            )

            # Step 4: Apply PPCM (no grad)
            with torch.no_grad():
                ppcm_feats = model.ppcm(
                    teacher_feats, depth, wt
                )

            # Step 5: Correction Header (with grad)
            corrected_feats = model.header(ppcm_feats)

            # Step 6: Distillation loss
            # Header should recover teacher features from PPCM-modulated
            loss = sum(
                F.mse_loss(cf, tf.detach())
                for cf, tf in zip(corrected_feats, teacher_feats)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.6f}")

        torch.save({
            'epoch':        epoch + 1,
            'header_state': model.header.state_dict(),
            'loss':         avg_loss
        }, os.path.join(WORK_DIR, f'epoch_{epoch+1}.pth'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch':        epoch + 1,
                'header_state': model.header.state_dict(),
                'loss':         avg_loss
            }, os.path.join(WORK_DIR, 'best.pth'))
            print(f"  → Best saved (loss={avg_loss:.6f})")

    print("Training complete.")


if __name__ == '__main__':
    print("=" * 50)
    print("PPCM + YOLOv8: Training")
    print("=" * 50)
    train()
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from iop_table import ALL_WATER_TYPES, BETA_D


# ── Water Type Estimator ────────────────────────────────────────────────
JERLOV_RGB_REFS = {
    'I':   (0.18, 0.28, 0.54),
    'II':  (0.22, 0.33, 0.45),
    'III': (0.28, 0.38, 0.34),
    '1C':  (0.30, 0.37, 0.33),
    '5C':  (0.33, 0.38, 0.29),
    '9C':  (0.36, 0.36, 0.28),
}

def estimate_water_type(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
    total = r + g + b + 1e-8
    ratios = (r/total, g/total, b/total)
    dists  = {
        wt: np.sqrt(sum((a-b)**2 for a,b in zip(ratios, ref)))
        for wt, ref in JERLOV_RGB_REFS.items()
    }
    return min(dists, key=dists.get)


# ── Depth Estimator (lazy load) ─────────────────────────────────────────
_depth_pipe = None

def get_depth_estimator(device='cuda'):
    global _depth_pipe
    if _depth_pipe is None:
        from transformers import pipeline as hf_pipeline
        _depth_pipe = hf_pipeline(
            task  = "depth-estimation",
            model = "depth-anything/Depth-Anything-V2-Small-hf",
            device = 0 if device == 'cuda' else -1
        )
        print("Depth estimator loaded.")
    return _depth_pipe

def estimate_depth(img_path, device='cuda'):
    """Returns normalized depth map (1, H, W) as torch tensor [0,1]."""
    from PIL import Image as PILImage
    pipe = get_depth_estimator(device)
    pil  = PILImage.open(img_path).convert('RGB')
    res  = pipe(pil)
    d    = np.array(res['depth']).astype(np.float32)
    dmin, dmax = d.min(), d.max()
    if dmax > dmin:
        d = (d - dmin) / (dmax - dmin)
    return torch.from_numpy(d).unsqueeze(0)  # (1, H, W)


# ── Dataset ─────────────────────────────────────────────────────────────
class UnderwaterDetDataset(Dataset):
    """
    Dataset for PPCM Pipeline.
    Returns: image, target, depth_map, water_type, img_path

    Depth maps are precomputed and cached to avoid
    re-running depth estimator at each epoch.
    """
    def __init__(
        self,
        ann_file,
        base_path,
        type_dirs,
        depth_cache_dir='depth_cache',
        device='cuda'
    ):
        self.coco           = COCO(ann_file)
        self.base_path      = base_path
        self.type_dirs      = type_dirs
        self.img_ids        = list(self.coco.imgs.keys())
        self.depth_cache_dir = depth_cache_dir
        self.device         = device
        os.makedirs(depth_cache_dir, exist_ok=True)

    def find_image(self, fname):
        for t in self.type_dirs:
            path = os.path.join(self.base_path, t, fname)
            if os.path.exists(path):
                return path
        return None

    def get_depth(self, img_path):
        """Get depth map — from cache if available, else compute and cache."""
        fname    = os.path.basename(img_path)
        cache_pt = os.path.join(self.depth_cache_dir, fname.replace('.jpg', '_depth.pt'))

        if os.path.exists(cache_pt):
            return torch.load(cache_pt)

        depth = estimate_depth(img_path, self.device)
        torch.save(depth, cache_pt)
        return depth

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.find_image(img_info['file_name'])

        # ── Load image ────────────────────────────────────────────────
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)

        # ── Annotations ───────────────────────────────────────────────
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

        # ── Water type + Depth ────────────────────────────────────────
        water_type = estimate_water_type(img_path)
        depth_map  = self.get_depth(img_path)        # (1, H, W)

        return img_tensor, target, depth_map, water_type, img_path


def collate_fn(batch):
    imgs, targets, depths, water_types, paths = zip(*batch)
    return list(imgs), list(targets), list(depths), list(water_types), list(paths)
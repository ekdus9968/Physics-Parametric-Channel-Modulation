import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from model   import PPCMPipeline
from dataset import UnderwaterDetDataset, collate_fn
from visualize import (
    save_stage1_visualization,
    save_stage2_visualization,
    save_detection_visualization,
    save_training_curve,
)

# ── Config ────────────────────────────────────────────────────────────────
BASE_PATH    = 'data/S-UODAC2020'
WORK_DIR     = 'work_dirs/ppcm_new'
NUM_CLASSES  = 5
NUM_EPOCHS   = 12
BATCH_SIZE   = 2
LR           = 0.005
ALPHA        = 0.5     # Stage 1 backscatter strength
DEPTH_SCALE  = 1.0     # Stage 2 spatial weight scale
VIZ_EVERY    = 50      # visualize every N batches
VIZ_EPOCH    = [1, 6, 12]  # visualize detection at these epochs
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CATEGORIES = {1: 'echinus', 2: 'starfish', 3: 'holothurian', 4: 'scallop'}

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, 'viz', 'stage1'), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, 'viz', 'stage2'), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, 'viz', 'detections'), exist_ok=True)


def train(use_stage1=True, use_stage2=True):
    mode = f"s1={use_stage1}_s2={use_stage2}"
    print(f"\n{'='*60}")
    print(f"Training PPCM Pipeline: {mode}")
    print(f"Alpha={ALPHA}, DepthScale={DEPTH_SCALE}")
    print(f"{'='*60}")

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = UnderwaterDetDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations', 'instances_source.json'),
        base_path = BASE_PATH,
        type_dirs = ['type1','type2','type3','type4','type5','type6'],
        depth_cache_dir = os.path.join(WORK_DIR, 'depth_cache'),
        device    = str(DEVICE)
    )
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0
    )
    print(f"Train samples: {len(dataset)}")

    # ── Model ──────────────────────────────────────────────────────────
    model = PPCMPipeline(
        num_classes = NUM_CLASSES,
        alpha       = ALPHA,
        depth_scale = DEPTH_SCALE,
        use_stage1  = use_stage1,
        use_stage2  = use_stage2
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.1
    )

    losses_per_epoch = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        viz_done_s1 = False
        viz_done_s2 = False
        viz_done_det = False

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')

        for batch_idx, (imgs, targets, depths, water_types, paths) in enumerate(pbar):

            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            depths  = [d.to(DEVICE) for d in depths]

            # ── Forward ──────────────────────────────────────────────
            loss_dict = model(
                images      = imgs,
                targets     = targets,
                depth_maps  = depths,
                water_types = water_types
            )
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'wt': water_types[0]
            })

            # ── Visualization ─────────────────────────────────────────
            if batch_idx % VIZ_EVERY == 0:

                # Stage 1 visualization
                if use_stage1 and 'stage1' in model.last_viz and not viz_done_s1:
                    path = save_stage1_visualization(
                        viz_dict  = model.last_viz['stage1'],
                        save_dir  = os.path.join(WORK_DIR, 'viz', 'stage1'),
                        epoch     = epoch,
                        batch_idx = batch_idx
                    )
                    print(f"\n  Stage1 viz → {path}")
                    viz_done_s1 = True

                # Stage 2 visualization
                if use_stage2 and 'stage2' in model.last_viz and not viz_done_s2:
                    path = save_stage2_visualization(
                        viz_dict  = model.last_viz['stage2'],
                        save_dir  = os.path.join(WORK_DIR, 'viz', 'stage2'),
                        epoch     = epoch,
                        batch_idx = batch_idx
                    )
                    print(f"\n  Stage2 viz → {path}")
                    viz_done_s2 = True

        # ── Detection visualization (eval mode, first batch of test) ──
        if epoch in VIZ_EPOCH:
            model.eval()
            sample_imgs = [imgs[0].detach()]
            sample_depths = [depths[0].detach()]
            sample_wts    = [water_types[0]]

            with torch.no_grad():
                preds = model(
                    images      = sample_imgs,
                    depth_maps  = sample_depths,
                    water_types = sample_wts
                )

            path = save_detection_visualization(
                images    = [imgs[0].detach().cpu()],
                outputs   = preds,
                targets   = [targets[0]],
                save_dir  = os.path.join(WORK_DIR, 'viz', 'detections'),
                epoch     = epoch,
                batch_idx = 0,
                categories = CATEGORIES
            )
            print(f"\n  Detection viz → {path}")
            model.train()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        losses_per_epoch.append(avg_loss)
        print(f"\nEpoch {epoch}: avg_loss = {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(WORK_DIR, f'{mode}_epoch_{epoch}.pth')
        torch.save({
            'epoch':             epoch,
            'model_state_dict':  model.state_dict(),
            'loss':              avg_loss,
            'use_stage1':        use_stage1,
            'use_stage2':        use_stage2,
            'alpha':             ALPHA,
            'depth_scale':       DEPTH_SCALE,
        }, ckpt_path)

    # Save training curve
    curve_path = save_training_curve(
        losses_per_epoch,
        os.path.join(WORK_DIR, 'viz')
    )
    print(f"\nTraining curve → {curve_path}")

    # Save loss log
    log_path = os.path.join(WORK_DIR, f'{mode}_losses.json')
    with open(log_path, 'w') as f:
        json.dump({'losses': losses_per_epoch, 'mode': mode}, f, indent=2)

    return losses_per_epoch


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-stage1', action='store_true')
    parser.add_argument('--no-stage2', action='store_true')
    args = parser.parse_args()

    train(
        use_stage1 = not args.no_stage1,
        use_stage2 = not args.no_stage2
    )
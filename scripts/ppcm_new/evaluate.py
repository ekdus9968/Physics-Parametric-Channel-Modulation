import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json

from model   import PPCMPipeline
from dataset import UnderwaterDetDataset, collate_fn

BASE_PATH   = 'data/S-UODAC2020'
WORK_DIR    = 'work_dirs/ppcm_new'
NUM_CLASSES = 5
ALPHA       = 0.5
DEPTH_SCALE = 1.0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_checkpoint(checkpoint_path, use_stage1, use_stage2, label=''):
    model = PPCMPipeline(
        num_classes = NUM_CLASSES,
        alpha       = ALPHA,
        depth_scale = DEPTH_SCALE,
        use_stage1  = use_stage1,
        use_stage2  = use_stage2
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(DEVICE)

    dataset = UnderwaterDetDataset(
        ann_file  = os.path.join(BASE_PATH, 'COCO_Annotations', 'instances_target.json'),
        base_path = BASE_PATH,
        type_dirs = ['type7'],
        depth_cache_dir = os.path.join(WORK_DIR, 'depth_cache_test'),
        device    = str(DEVICE)
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    results = []
    with torch.no_grad():
        for imgs, _, depths, water_types, _ in tqdm(loader, desc=label):
            imgs   = [img.to(DEVICE) for img in imgs]
            depths = [d.to(DEVICE) for d in depths]

            outputs = model(
                images      = imgs,
                depth_maps  = depths,
                water_types = water_types
            )

            for img_idx, output in enumerate(outputs):
                # Get image_id from dataset
                img_id = dataset.coco.imgs[dataset.img_ids[
                    loader.dataset.img_ids.index(
                        dataset.img_ids[loader.dataset.img_ids.index(
                            dataset.img_ids[img_idx]
                        )]
                    )
                ]]
                # Simpler approach: track index
                pass

    # Re-run with proper id tracking
    results = []
    coco_gt = dataset.coco

    with torch.no_grad():
        for i, (imgs, targets, depths, water_types, _) in enumerate(
                tqdm(DataLoader(dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_fn, num_workers=0),
                     desc=label)):

            imgs   = [img.to(DEVICE) for img in imgs]
            depths = [d.to(DEVICE) for d in depths]

            outputs = model(
                images      = imgs,
                depth_maps  = depths,
                water_types = water_types
            )

            for target, output in zip(targets, outputs):
                img_id = target['image_id'].item()
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

    if not results:
        print(f"{label}: No detections")
        return 0.0, 0.0

    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[1] * 100, coco_eval.stats[0] * 100  # mAP@50, mAP@50:95


if __name__ == '__main__':
    print("="*60)
    print("PPCM New Pipeline — Ablation Evaluation")
    print("="*60)

    configs = [
        ('s1=False_s2=False', False, False, 'Baseline (no PPCM)'),
        ('s1=True_s2=False',  True,  False, 'Stage 1 only'),
        ('s1=False_s2=True',  False, True,  'Stage 2 only'),
        ('s1=True_s2=True',   True,  True,  'Full PPCM (S1+S2)'),
    ]

    all_results = {}

    for mode, s1, s2, label in configs:
        ckpt = os.path.join(WORK_DIR, f'{mode}_epoch_12.pth')
        if not os.path.exists(ckpt):
            print(f"\nCheckpoint not found: {ckpt}")
            continue

        print(f"\n{'-'*40}")
        print(f"Evaluating: {label}")
        mAP50, mAP5095 = evaluate_checkpoint(ckpt, s1, s2, label)
        all_results[label] = {'mAP50': mAP50, 'mAP50_95': mAP5095}
        print(f"  mAP@50:    {mAP50:.2f}%")
        print(f"  mAP@50:95: {mAP5095:.2f}%")

    print("\n" + "="*60)
    print("Ablation Summary")
    print("="*60)
    baseline = all_results.get('Baseline (no PPCM)', {}).get('mAP50', 0)
    for label, res in all_results.items():
        delta = res['mAP50'] - baseline
        marker = ' ← best' if res['mAP50'] == max(
            r['mAP50'] for r in all_results.values()) else ''
        print(f"{label:<30} mAP@50={res['mAP50']:.2f}%  ({delta:+.2f}%){marker}")

    with open(os.path.join(WORK_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {WORK_DIR}/ablation_results.json")
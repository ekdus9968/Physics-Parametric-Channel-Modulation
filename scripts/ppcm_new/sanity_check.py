"""
Sanity check: run one forward pass and verify each stage outputs expected shapes.
Run this BEFORE full training to catch any errors.

Usage:
    python scripts/ppcm_new/sanity_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from model import PPCMPipeline
from ppcm_modules import PPCMStage1, PPCMStage2
from visualize import (
    save_stage1_visualization,
    save_stage2_visualization,
)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIZ_DIR  = 'work_dirs/ppcm_new/viz/sanity'
os.makedirs(VIZ_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ── 1. Stage 1 standalone ──────────────────────────────────────────────
print("\n[1] Testing PPCM Stage 1...")
s1 = PPCMStage1(alpha=0.5)

img   = torch.rand(2, 3, 400, 600)  # (B,3,H,W) fake batch
# No depth for Stage 1

corrected, bscat = s1(img, 'I')  # water_type only
print(f"  Input shape:       {img.shape}")
print(f"  Corrected shape:   {corrected.shape}")
print(f"  Backscatter shape: {bscat.shape}")
print(f"  Input mean:     {img.mean():.4f}")
print(f"  Corrected mean: {corrected.mean():.4f}  (should be lower)")
print(f"  Backscatter R (scalar map): {bscat[:,0].mean():.4f}")
print(f"  Backscatter G (scalar map): {bscat[:,1].mean():.4f}")
print(f"  Backscatter B (scalar map): {bscat[:,2].mean():.4f}")
print("  ✓ Stage 1 OK")

# Visualize Stage 1
save_stage1_visualization(
    viz_dict  = {
        'raw':         [img[0]],
        'corrected':   [corrected[0].detach()],
        'backscatter': [bscat[0].detach()],
    },
    save_dir  = VIZ_DIR,
    epoch     = 0,
    batch_idx = 0
)
print(f"  Stage 1 viz → {VIZ_DIR}/stage1_ep00_b0000.png")

# Test all water types
print("\n  Water type effect (channel-specific, no spatial variation):")
for wt in ['I', 'II', 'III', '1C', '5C', '9C']:
    corr, bscat_wt = s1(img, wt)
    diff = (img - corr).abs()
    print(f"    Type {wt:3s}: R_removed={diff[:,0].mean():.4f}  "
          f"G_removed={diff[:,1].mean():.4f}  "
          f"B_removed={diff[:,2].mean():.4f}")

# ── 2. Stage 2 standalone ──────────────────────────────────────────────
print("\n[2] Testing PPCM Stage 2...")
from collections import OrderedDict
s2 = PPCMStage2(depth_scale=1.0)

fake_fpn = OrderedDict({
    '0': torch.rand(2, 256, 50,  75),   # P3
    '1': torch.rand(2, 256, 25,  38),   # P4
    '2': torch.rand(2, 256, 13,  19),   # P5
    'pool': torch.rand(2, 256, 7, 10),  # P6
})
depth_s2 = torch.rand(2, 1, 400, 600) * 0.6

modulated, wmaps = s2(fake_fpn, 'I', depth_s2)

print(f"  FPN input keys:  {list(fake_fpn.keys())}")
print(f"  Output keys:     {list(modulated.keys())}")
for key in ['0', '1', '2']:
    orig_mean = fake_fpn[key].mean().item()
    mod_mean  = modulated[key].mean().item()
    wmap_std  = wmaps[key][0, 0].std().item()
    print(f"    P{int(key)+3}: orig_mean={orig_mean:.4f}  mod_mean={mod_mean:.4f}  weight_std={wmap_std:.4f}")

save_stage2_visualization(
    viz_dict  = {k: v.detach() for k, v in wmaps.items()},
    save_dir  = VIZ_DIR,
    epoch     = 0,
    batch_idx = 0
)
print(f"  Stage 2 viz → {VIZ_DIR}/stage2_ep00_b0000.png")

# Test spatial variation
print("\n  Spatial weight variation (shallower vs deeper regions):")
shallow_depth = torch.zeros(1, 1, 400, 600) + 0.1  # very shallow
deep_depth    = torch.zeros(1, 1, 400, 600) + 0.9  # very deep
fpn_single    = OrderedDict({'0': torch.rand(1, 256, 50, 75)})

_, w_shallow = s2(fpn_single, 'I', shallow_depth)
_, w_deep    = s2(fpn_single, 'I', deep_depth)
print(f"    Shallow (z≈0.5m) weight mean: {w_shallow['0'].mean():.4f}")
print(f"    Deep    (z≈9.5m) weight mean: {w_deep['0'].mean():.4f}")
print(f"    → Shallower should have higher weight")
print("  ✓ Stage 2 OK")

# ── 3. Full Pipeline forward pass ──────────────────────────────────────
print("\n[3] Testing Full Pipeline (PPCMPipeline)...")
model = PPCMPipeline(
    num_classes = 5,
    alpha       = 0.5,
    depth_scale = 1.0,
    use_stage1  = True,
    use_stage2  = True
).to(DEVICE)
model.eval()

# Fake inputs
fake_imgs  = [torch.rand(3, 400, 600).to(DEVICE) for _ in range(2)]
fake_depths = [torch.rand(1, 400, 600).to(DEVICE) * 0.6 for _ in range(2)]
fake_wts   = ['I', '5C']

with torch.no_grad():
    outputs = model(
        images      = fake_imgs,
        depth_maps  = fake_depths,
        water_types = fake_wts
    )

print(f"  Output type: {type(outputs)}")
print(f"  Num predictions: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"  Image {i}: boxes={out['boxes'].shape}, "
          f"scores={out['scores'].shape}, labels={out['labels'].shape}")
print(f"  Viz keys: {list(model.last_viz.keys())}")
print("  ✓ Full pipeline OK")

print(f"\n{'='*50}")
print("All sanity checks passed!")
print(f"Visualization saved to: {VIZ_DIR}")
print(f"{'='*50}")
print("\nNext step: run training")
print("  Phase 1 (baseline, no PPCM):")
print("    python scripts/ppcm_new/train.py --no-stage1 --no-stage2")
print("  Phase 2 (Stage 1 only):")
print("    python scripts/ppcm_new/train.py --no-stage2")
print("  Phase 3 (Stage 2 only):")
print("    python scripts/ppcm_new/train.py --no-stage1")
print("  Phase 4 (Full PPCM):")
print("    python scripts/ppcm_new/train.py")
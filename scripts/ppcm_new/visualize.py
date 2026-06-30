import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2


def tensor_to_numpy(t):
    """Convert (3,H,W) or (1,H,W) tensor [0,1] to numpy uint8."""
    t = t.detach().cpu().float()
    if t.shape[0] == 3:
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = t.squeeze(0).numpy()
    return np.clip(arr, 0, 1)


def save_stage1_visualization(viz_dict, save_dir, epoch, batch_idx, max_images=2):
    """
    Visualize Stage 1: raw vs corrected vs backscatter map.
    3 columns per image: [Raw] [Corrected] [Backscatter Removed]
    """
    os.makedirs(save_dir, exist_ok=True)

    raws       = viz_dict['raw']
    corrected  = viz_dict['corrected']
    backscatter = viz_dict['backscatter']

    n = min(len(raws), max_images)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        raw_np  = tensor_to_numpy(raws[i])
        corr_np = tensor_to_numpy(corrected[i])
        bscat_np = tensor_to_numpy(backscatter[i])
        diff_np = np.abs(raw_np - corr_np)

        axes[i, 0].imshow(raw_np)
        axes[i, 0].set_title(f'Raw Image\nmean={raw_np.mean():.3f}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(corr_np)
        axes[i, 1].set_title(f'After Stage 1\nmean={corr_np.mean():.3f}', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(bscat_np)
        axes[i, 2].set_title(f'Backscatter Map\nmean={bscat_np.mean():.3f}', fontsize=10)
        axes[i, 2].axis('off')

        # Difference: what was removed
        im = axes[i, 3].imshow(diff_np, cmap='hot')
        axes[i, 3].set_title(f'|Raw - Corrected|\nmax={diff_np.max():.3f}', fontsize=10)
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3])

    plt.suptitle(f'PPCM Stage 1 — Epoch {epoch}, Batch {batch_idx}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f'stage1_ep{epoch:02d}_b{batch_idx:04d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_stage2_visualization(viz_dict, save_dir, epoch, batch_idx):
    """
    Visualize Stage 2 spatial weight maps for P3, P4, P5.
    Shows how different spatial regions are weighted.
    """
    os.makedirs(save_dir, exist_ok=True)

    keys = sorted([k for k in viz_dict.keys() if k in ['0', '1', '2']])
    level_names = {'0': 'P3 (small)', '1': 'P4 (medium)', '2': 'P5 (large)'}

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        wmap = viz_dict[key]  # (B,1,H,W)
        w    = wmap[0, 0].cpu().numpy()  # first in batch

        im = ax.imshow(w, cmap='RdYlGn', vmin=0, vmax=2)
        ax.set_title(
            f'{level_names.get(key, key)}\n'
            f'res={w.shape[0]}×{w.shape[1]}\n'
            f'mean={w.mean():.3f}, std={w.std():.3f}',
            fontsize=10
        )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        f'PPCM Stage 2 Spatial Weight Maps\n'
        f'Green=reliable (shallow), Red=unreliable (deep)\n'
        f'Epoch {epoch}, Batch {batch_idx}',
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f'stage2_ep{epoch:02d}_b{batch_idx:04d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_detection_visualization(images, outputs, targets, save_dir,
                                  epoch, batch_idx,
                                  categories, max_images=2,
                                  score_thresh=0.3):
    """
    Visualize detection results: GT boxes (green) vs predicted boxes (red).
    """
    os.makedirs(save_dir, exist_ok=True)

    n = min(len(images), max_images)
    fig, axes = plt.subplots(1, n, figsize=(12 * n, 8))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= len(images):
            break

        img_np = tensor_to_numpy(images[i])
        ax.imshow(img_np)

        # GT boxes — green
        if targets and i < len(targets):
            for box, label in zip(
                    targets[i]['boxes'].cpu().numpy(),
                    targets[i]['labels'].cpu().numpy()):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)
                cat = categories.get(int(label), str(label))
                ax.text(x1, y1-2, f'GT:{cat}', color='lime',
                        fontsize=8, backgroundcolor='black')

        # Predicted boxes — red
        if outputs and i < len(outputs):
            for box, score, label in zip(
                    outputs[i]['boxes'].cpu().numpy(),
                    outputs[i]['scores'].cpu().numpy(),
                    outputs[i]['labels'].cpu().numpy()):
                if score < score_thresh:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                cat = categories.get(int(label), str(label))
                ax.text(x1, y2+2, f'{cat}:{score:.2f}',
                        color='red', fontsize=8, backgroundcolor='black')

        ax.axis('off')
        ax.set_title(f'Image {i} | Green=GT  Red=Pred (>{score_thresh:.0%})')

    plt.suptitle(f'Detections — Epoch {epoch}, Batch {batch_idx}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f'det_ep{epoch:02d}_b{batch_idx:04d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_training_curve(losses_per_epoch, save_dir):
    """Plot training loss curve."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_per_epoch, marker='o', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, 'training_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path
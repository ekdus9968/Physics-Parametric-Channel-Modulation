import torch
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
import os

# Config
WORK_DIR    = 'work_dirs/baseline'
OUTPUT_DIR  = 'work_dirs/ppcm_stage1'
NUM_CLASSES = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IOP table: beta_D per channel per Jerlov type
# Source: Solonenko & Mobley, Applied Optics 2015
IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

def load_model(checkpoint_path):
    """Load pretrained Faster R-CNN."""
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def extract_w1_sensitivity(model):
    """
    Extract RGB sensitivity from Conv1 weights.
    W1: (64, 3, 7, 7)
    S[i, c] = Frobenius norm of W1[i, c, :, :]
    Returns S_normalized: (64, 3) softmax normalized
    """
    # ResNet Conv1 is backbone.body.layer0 or backbone.body.conv1
    w1 = model.backbone.body.conv1.weight.data  # (64, 3, 7, 7)
    print(f"W1 shape: {w1.shape}")

    # Sensitivity: how much each output channel responds to each RGB input
    S = torch.norm(w1.view(64, 3, -1), dim=2)  # (64, 3)
    S_normalized = F.softmax(S, dim=1)          # normalize across RGB

    print(f"S shape: {S.shape}")
    print(f"\nMean sensitivity per RGB channel:")
    print(f"  R: {S_normalized[:, 0].mean():.4f}")
    print(f"  G: {S_normalized[:, 1].mean():.4f}")
    print(f"  B: {S_normalized[:, 2].mean():.4f}")

    return S_normalized

def compute_channel_weights(S_normalized, water_type):
    """
    Compute feature channel weights from physics.
    reliability_c = 1 / beta_D_c(t)  (lower attenuation = more reliable)
    channel_weight_i = sum_c S[i,c] * reliability_c
    """
    beta = IOP_TABLE[water_type]

    # Channel reliability: inverse of attenuation
    reliability = torch.tensor([
        1.0 / beta['R'],
        1.0 / beta['G'],
        1.0 / beta['B']
    ])

    # Normalize reliability
    reliability = reliability / reliability.sum()

    # Feature channel weights: (64,)
    channel_weights = S_normalized @ reliability  # (64,)

    # Normalize to [0, 1]
    channel_weights = channel_weights / channel_weights.max()

    return channel_weights

def visualize_channel_weights():
    """Compare channel weights across water types."""
    checkpoint_path = os.path.join(WORK_DIR, 'epoch_12.pth')
    model = load_model(checkpoint_path)
    S_normalized = extract_w1_sensitivity(model)

    water_types = ['I', 'II', 'III', '1C', '5C', '9C']
    all_weights  = {}

    print("\n=== Channel Weight Statistics per Water Type ===")
    for wt in water_types:
        weights = compute_channel_weights(S_normalized, wt)
        all_weights[wt] = weights.numpy()
        print(f"Type {wt:3s}: mean={weights.mean():.4f}, "
              f"std={weights.std():.4f}, "
              f"min={weights.min():.4f}, "
              f"max={weights.max():.4f}")

    # Plot: channel weight distribution per water type
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Channel Weights by Water Type (PPCM Stage 1)',
                 fontsize=14)

    for idx, wt in enumerate(water_types):
        ax = axes[idx // 3][idx % 3]
        weights = all_weights[wt]
        ax.bar(range(64), weights, color='steelblue', alpha=0.7)
        ax.set_title(f'Jerlov Type {wt}')
        ax.set_xlabel('Feature Channel Index')
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=weights.mean(), color='red',
                   linestyle='--', label=f'mean={weights.mean():.3f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'channel_weights_per_type.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    # Plot: Type I vs Type 9C comparison
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(64)
    ax.bar(x - 0.2, all_weights['I'],  width=0.4,
           label='Type I (clear blue)', alpha=0.7, color='blue')
    ax.bar(x + 0.2, all_weights['9C'], width=0.4,
           label='Type 9C (turbid)',    alpha=0.7, color='brown')
    ax.set_title('Type I vs Type 9C Channel Weights')
    ax.set_xlabel('Feature Channel Index')
    ax.set_ylabel('Weight')
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'type_I_vs_9C.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    print("=" * 50)
    print("Phase 3: PPCM Stage 1 - W1 Bridge Inspection")
    print("=" * 50)
    visualize_channel_weights()
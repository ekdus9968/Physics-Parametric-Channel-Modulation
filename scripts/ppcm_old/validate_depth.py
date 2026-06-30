import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import pipeline

# Config
BASE_PATH = 'data/S-UODAC2020'
OUTPUT_DIR = 'work_dirs/depth_validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Load Depth Anything V2
print("Loading Depth Anything V2...")
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if DEVICE == 'cuda' else -1
)
print("Model loaded.")

def estimate_depth(img_path):
    """Estimate depth map from image."""
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    result = depth_estimator(img)
    depth = np.array(result['depth'])
    return depth

def visualize_sample(type_name, n_samples=3):
    """Visualize depth maps for sample images from a type."""
    type_path = os.path.join(BASE_PATH, type_name)
    images = [f for f in os.listdir(type_path)
              if f.endswith('.jpg')][:n_samples]

    fig, axes = plt.subplots(n_samples, 2,
                             figsize=(10, 4 * n_samples))
    fig.suptitle(f'{type_name} - RGB vs Depth', fontsize=14)

    for i, fname in enumerate(images):
        img_path = os.path.join(type_path, fname)

        # RGB image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Depth map
        depth = estimate_depth(img_path)

        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'RGB: {fname}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(depth, cmap='plasma')
        axes[i, 1].set_title('Depth Map')
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{type_name}_depth.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    print("=" * 50)
    print("Phase 2: Depth Estimator Validation")
    print("=" * 50)

    # Validate on 3 types
    for type_name in ['type1', 'type6', 'type7']:
        print(f"\nProcessing {type_name}...")
        visualize_sample(type_name, n_samples=3)

    print("\nDone. Check work_dirs/depth_validation/")
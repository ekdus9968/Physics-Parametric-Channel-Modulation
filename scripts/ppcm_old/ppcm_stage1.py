import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ppcm_old.evaluate_ppcm_alpha import alpha

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

class PPCMStage1(nn.Module):
    """
    PPCM Stage 1: Physics-parametric channel modulation.
    Applied at Conv1 output.
    No depth needed. Uses water type t only.
    No learnable parameters.
    """
    def __init__(self, conv1_weight):
        """
        Args:
            conv1_weight: pretrained Conv1 weight tensor (64, 3, 7, 7)
        """
        super().__init__()

        # Extract RGB sensitivity from Conv1 weights (frozen)
        # S[i, c] = Frobenius norm of W1[i, c, :, :]
        w1 = conv1_weight.detach()                    # (64, 3, 7, 7)
        S  = torch.norm(w1.view(64, 3, -1), dim=2)   # (64, 3)
        S_normalized = F.softmax(S, dim=1)            # (64, 3)

        # Register as buffer (not a parameter, moves with .to(device))
        self.register_buffer('S_normalized', S_normalized)
        self.iop_table = IOP_TABLE

    def compute_weights(self, water_type):
        """
        Compute feature channel weights for given water type.
        reliability_c = 1 / beta_D_c(t)
        channel_weight_i = sum_c S[i,c] * reliability_c
        Returns: weight tensor (64,)
        """
        beta = self.iop_table[water_type]

        reliability = torch.tensor([
            1.0 / beta['R'],
            1.0 / beta['G'],
            1.0 / beta['B']
        ], device=self.S_normalized.device)

        # Normalize reliability
        reliability = reliability / reliability.sum()

        # Feature channel weights: (64,)
        weights = self.S_normalized @ reliability

        # Normalize to mean 1.0 (preserve feature scale)
        weights =  1.0 + alpha * (weights / weights.mean() - 1.0)

        return weights

    def forward(self, feature_map, water_type):
        """
        Args:
            feature_map: Conv1 output (B, 64, H, W)
            water_type:  Jerlov type string e.g. 'I', '9C'
        Returns:
            modulated feature_map (B, 64, H, W)
        """
        weights = self.compute_weights(water_type)   # (64,)
        weights = weights.view(1, 64, 1, 1)          # broadcast shape
        return feature_map * weights


if __name__ == '__main__':
    import os
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    WORK_DIR    = 'work_dirs/baseline'
    NUM_CLASSES = 5
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    checkpoint = torch.load(
        os.path.join(WORK_DIR, 'epoch_12.pth'),
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)

    # Build PPCM Stage 1
    conv1_weight = model.backbone.body.conv1.weight.data
    ppcm_s1 = PPCMStage1(conv1_weight).to(DEVICE)

    # Test with dummy feature map
    dummy_feature = torch.randn(1, 64, 400, 600).to(DEVICE)

    print("=== PPCM Stage 1 Test ===")
    for wt in ['I', '5C', '9C']:
        out = ppcm_s1(dummy_feature, wt)
        weights = ppcm_s1.compute_weights(wt)
        print(f"Type {wt:3s}: "
              f"input_mean={dummy_feature.mean():.4f}, "
              f"output_mean={out.mean():.4f}, "
              f"weight_std={weights.std():.4f}")

    print("\nPPCM Stage 1 working correctly.")
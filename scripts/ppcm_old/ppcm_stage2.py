import torch
import torch.nn as nn
import torch.nn.functional as F

# IOP table
IOP_TABLE = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

# Dominant attenuation per water type
# Use mean of R,G,B beta_D as spatial weighting coefficient
def get_dominant_beta(water_type):
    beta = IOP_TABLE[water_type]
    return (beta['R'] + beta['G'] + beta['B']) / 3.0

# Depth scale factor
# Depth Anything V2 gives relative depth [0,1]
# Scale to approximate underwater range [0.5m, 10m]
DEPTH_MIN = 0.5
DEPTH_MAX = 10.0

class PPCMStage2(nn.Module):
    """
    PPCM Stage 2: Physics-parametric spatial modulation.
    Applied at FPN P3, P4 output.
    Uses water type t + depth map z(x).
    No learnable parameters.
    """
    def __init__(self):
        super().__init__()
        self.iop_table = IOP_TABLE

    def scale_depth(self, depth_map):
        """
        Scale relative depth [0,1] to absolute [0.5m, 10m].
        depth_map: (1, H, W) normalized
        """
        return DEPTH_MIN + depth_map * (DEPTH_MAX - DEPTH_MIN)

    def compute_spatial_weight(self, depth_map, water_type, target_size):
        """
        Compute per-pixel spatial reliability.
        weight(x) = exp(-beta_dominant(t) * z(x))
        
        Args:
            depth_map:   (1, 1, H, W) relative depth
            water_type:  Jerlov type string
            target_size: (H_target, W_target) for FPN level
        Returns:
            spatial_weight: (1, 1, H_target, W_target)
        """
        # Scale depth to absolute
        z = self.scale_depth(depth_map)

        # Downsample depth to FPN resolution
        z_resized = F.interpolate(
            z,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Dominant beta for this water type
        beta = get_dominant_beta(water_type)

        # Spatial reliability: exp(-beta * z)
        spatial_weight = torch.exp(-beta * z_resized)

        # Normalize to mean 1.0 (preserve feature scale)
        spatial_weight = spatial_weight / spatial_weight.mean()

        return spatial_weight

    def forward(self, fpn_features, depth_map, water_type):
        """
        Apply spatial weighting to FPN P3, P4 features.

        Args:
            fpn_features: dict with keys '0'(P3), '1'(P4), '2'(P5)...
                         from FPN output
            depth_map:   (1, 1, H, W) relative depth from estimator
            water_type:  Jerlov type string
        Returns:
            modulated fpn_features dict
        """
        modulated = {}
        for key, feat in fpn_features.items():
            # Only apply to P3 (key='0') and P4 (key='1')
            if key in ['0', '1']:
                H, W = feat.shape[2], feat.shape[3]
                spatial_weight = self.compute_spatial_weight(
                    depth_map, water_type, (H, W)
                )
                modulated[key] = feat * spatial_weight
            else:
                modulated[key] = feat
        return modulated


if __name__ == '__main__':
    # Quick test
    ppcm_s2 = PPCMStage2()

    # Dummy FPN features
    fpn_features = {
        '0': torch.randn(1, 256, 100, 150),  # P3
        '1': torch.randn(1, 256, 50,  75),   # P4
        '2': torch.randn(1, 256, 25,  38),   # P5
    }

    # Dummy depth map
    depth_map = torch.rand(1, 1, 800, 1408)

    print("=== PPCM Stage 2 Test ===")
    for wt in ['I', '5C', '9C']:
        out = ppcm_s2(fpn_features, depth_map, wt)
        beta = get_dominant_beta(wt)
        print(f"Type {wt:3s}: beta_dominant={beta:.3f}, "
              f"P3_mean_before={fpn_features['0'].mean():.4f}, "
              f"P3_mean_after={out['0'].mean():.4f}")

    print("\nPPCM Stage 2 working correctly.")
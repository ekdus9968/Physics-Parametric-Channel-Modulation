import torch
import torch.nn as nn
import torch.nn.functional as F
from iop_table import BETA_D, BETA_B, B_INF, DEPTH_MIN, DEPTH_MAX, get_dominant_beta


class PPCMStage1(nn.Module):
    """
    Stage 1: Pixel-space backscatter subtraction.

    Applied BEFORE backbone on raw [0,1] image.
    BN cannot absorb this — changes actual pixel values,
    not feature activations.

    Physics:
        B_est_c(x) = B_inf_c(t) * (1 - exp(-beta_B_c(t) * z(x)))
        corrected_c(x) = clamp(raw_c(x) - alpha * B_est_c(x), 0, 1)

    No learnable parameters.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, img, water_type, depth_map):
        """
        img:       (B, 3, H, W)  raw pixel values [0, 1]
        water_type: string e.g. 'I', '9C'
        depth_map: (B, 1, H, W)  relative depth [0, 1]

        Returns:
            corrected: (B, 3, H, W)  backscatter-subtracted image [0, 1]
            backscatter_map: (B, 3, H, W)  estimated backscatter (for visualization)
        """
        b_inf  = B_INF[water_type]
        beta_b = BETA_B[water_type]

        # Scale relative depth to absolute [DEPTH_MIN, DEPTH_MAX]
        z = DEPTH_MIN + depth_map * (DEPTH_MAX - DEPTH_MIN)  # (B,1,H,W)

        corrected    = img.clone()
        b_est_maps   = []

        for i, c in enumerate(['R', 'G', 'B']):
            # Per-pixel backscatter estimate
            B_est = b_inf[c] * (1.0 - torch.exp(-beta_b[c] * z))  # (B,1,H,W)
            b_est_maps.append(B_est)

            # Subtract backscatter with strength alpha
            corrected[:, i:i+1] = torch.clamp(
                img[:, i:i+1] - self.alpha * B_est, 0.0, 1.0
            )

        backscatter_map = torch.cat(b_est_maps, dim=1)  # (B, 3, H, W)
        return corrected, backscatter_map


class PPCMStage2(nn.Module):
    """
    Stage 2: FPN-output spatial reliability weighting.

    Applied AFTER FPN, BEFORE detection head.
    FPN output → Detection Head has NO BN layer.
    Therefore weighting is NOT absorbed.

    Physics:
        spatial_weight(x) = exp(-beta_D_dominant(t) * scale * z(x))
        normalized to mean=1 to preserve feature scale
        P_k' = P_k * spatial_weight_k

    No learnable parameters.
    """
    def __init__(self, depth_scale=1.0):
        super().__init__()
        self.depth_scale = depth_scale

    def forward(self, fpn_features, water_type, depth_map):
        """
        fpn_features: OrderedDict {'0': P3, '1': P4, '2': P5, '3': P6, 'pool': P7}
                      P3: (B, 256, H/8,  W/8)
                      P4: (B, 256, H/16, W/16)
                      P5: (B, 256, H/32, W/32)
        water_type:   string
        depth_map:    (B, 1, H, W) relative depth [0,1]

        Returns:
            modulated_features: same structure as fpn_features
            weight_maps: dict of spatial weight maps (for visualization)
        """
        beta = get_dominant_beta(water_type)
        z    = DEPTH_MIN + depth_map * (DEPTH_MAX - DEPTH_MIN)  # (B,1,H,W)

        modulated  = {}
        weight_maps = {}

        for key, feat in fpn_features.items():
            H, W = feat.shape[2], feat.shape[3]

            # Resize depth to FPN level resolution
            z_resized = F.interpolate(
                z, size=(H, W),
                mode='bilinear', align_corners=False
            )

            # Physics spatial reliability
            weight = torch.exp(-beta * self.depth_scale * z_resized)

            # Normalize around 1.0 to preserve feature scale
            weight = weight / (weight.mean() + 1e-8)

            modulated[key]   = feat * weight
            weight_maps[key] = weight.detach()

        return modulated, weight_maps
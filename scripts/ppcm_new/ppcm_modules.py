import torch
import torch.nn as nn
import torch.nn.functional as F
from iop_table import BETA_D, BETA_B, B_INF, DEPTH_MIN, DEPTH_MAX, get_dominant_beta


class PPCMStage1(nn.Module):
    """
    Stage 1: Pixel-space channel correction using water type ONLY.
    No depth used here — depth is Stage 2's responsibility.

    Physics factorization:
        I^c(x) = J^c(x) · exp(-β_D_c(t)·z(x))  +  B_∞(t)·(1-exp(-β_B(t)·z(x)))
                      channel term: t only              spatial term: z(x) only
                           ↓                                    ↓
                        Stage 1                              Stage 2

    Stage 1 uses the channel term only.
    β_D_c(t) tells us how much each channel is attenuated by water type t.
    High β_D_c → channel is unreliable → subtract more from that channel.

    Global (non-spatial) backscatter estimate per channel:
        B_global_c = B_inf_c(t) · (1 - exp(-β_D_c(t)))
        corrected_c = clamp(raw_c - alpha · B_global_c, 0, 1)

    Applied BEFORE backbone. BN cannot absorb — changes actual pixel values.
    No learnable parameters.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, img, water_type):
        """
        img:        (B, 3, H, W)  raw pixel values [0, 1]
        water_type: string e.g. 'I', '9C'
        NO depth — water type determines channel-level correction only.

        Example (Type I):
          β_D_R = 0.345 → reliability = exp(-0.345) = 0.708 → B_global_R = 0.12 × 0.292 = 0.035
          β_D_B = 0.017 → reliability = exp(-0.017) = 0.983 → B_global_B = 0.15 × 0.017 = 0.003
          → R channel corrected more than B channel
          → R channel was less reliable under Type I (heavy attenuation)

        Example (Type 9C):
          β_D_R = 0.290 → B_global_R = 0.22 × 0.252 = 0.055
          β_D_B = 0.349 → B_global_B = 0.09 × 0.295 = 0.027
          → B channel now also corrected significantly (reversal)

        Returns:
            corrected:       (B, 3, H, W)  channel-corrected image [0, 1]
            backscatter_map: (B, 3, H, W)  per-channel correction amount
        """
        b_inf  = B_INF[water_type]   # background light per channel
        beta_d = BETA_D[water_type]  # direct attenuation per channel

        corrected   = img.clone()
        b_est_maps  = []

        for i, c in enumerate(['R', 'G', 'B']):
            # Channel reliability: how much signal survives under this water type
            # Higher β_D_c → more attenuation → lower reliability
            reliability = torch.exp(torch.tensor(-beta_d[c], dtype=torch.float32))

            # Global backscatter estimate for this channel (scalar, no spatial variation)
            # More attenuation → less reliable → more backscatter to subtract
            B_global = b_inf[c] * (1.0 - reliability)  # scalar

            b_est_maps.append(
                torch.full_like(img[:, i:i+1], B_global)  # (B,1,H,W) constant map
            )

            # Subtract global backscatter with strength alpha
            corrected[:, i:i+1] = torch.clamp(
                img[:, i:i+1] - self.alpha * B_global,
                0.0, 1.0
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
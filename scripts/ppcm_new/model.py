import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from ppcm_modules import PPCMStage1, PPCMStage2


class PPCMPipeline(nn.Module):
    """
    Full PPCM Pipeline:
    [Raw Image]
        → [PPCM Stage 1: pixel-space backscatter removal]
        → [ResNet50 Backbone]
        → [FPN]
        → [PPCM Stage 2: spatial reliability weighting]
        → [Detection Head: RPN + ROI Head]
        → [Predictions]

    Trained end-to-end. No mismatch between training and inference.

    Args:
        num_classes: number of detection categories + 1 (background)
        alpha:       Stage 1 backscatter removal strength [0,1]
        depth_scale: Stage 2 depth weighting scale factor
        use_stage1:  enable/disable Stage 1 for ablation
        use_stage2:  enable/disable Stage 2 for ablation
    """
    def __init__(
        self,
        num_classes=5,
        alpha=0.5,
        depth_scale=1.0,
        use_stage1=True,
        use_stage2=True
    ):
        super().__init__()

        # ── Build base FasterRCNN ──────────────────────────────────────
        base = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.detector = base

        # ── PPCM Modules ──────────────────────────────────────────────
        self.ppcm_s1    = PPCMStage1(alpha=alpha) if use_stage1 else None
        self.ppcm_s2    = PPCMStage2(depth_scale=depth_scale) if use_stage2 else None
        self.use_stage1 = use_stage1
        self.use_stage2 = use_stage2

        # Storage for visualization (populated during forward)
        self.last_viz = {}

    def forward(self, images, targets=None, depth_maps=None, water_types=None):
        """
        images:      list of (3, H, W) tensors [0,1]
        targets:     list of dicts with 'boxes', 'labels' (training only)
        depth_maps:  list of (1, H, W) tensors [0,1]
        water_types: list of strings e.g. ['I', '5C', ...]

        Returns (training):   loss dict
        Returns (inference):  list of detection dicts
        """
        self.last_viz = {}

        # ── PPCM Stage 1: pixel-space channel correction (NO depth) ──
        if self.use_stage1 and water_types is not None:
            corrected_images   = []
            s1_viz_raw         = []
            s1_viz_corrected   = []
            s1_viz_backscatter = []

            for img, wt in zip(images, water_types):
                raw_4d = img.unsqueeze(0)   # (1,3,H,W)

                # Stage 1 uses water_type only — no depth
                corrected, bscat = self.ppcm_s1(raw_4d, wt)

                corrected_images.append(corrected.squeeze(0))
                s1_viz_raw.append(img.detach().cpu())
                s1_viz_corrected.append(corrected.squeeze(0).detach().cpu())
                s1_viz_backscatter.append(bscat.squeeze(0).detach().cpu())

            images = corrected_images

            self.last_viz['stage1'] = {
                'raw':         s1_viz_raw,
                'corrected':   s1_viz_corrected,
                'backscatter': s1_viz_backscatter,
            }
        else:
            corrected_images = images

        # ── PPCM Stage 2: FPN spatial weighting ───────────────────────
        # Hook on backbone to intercept FPN output
        # Per-image processing to handle variable image sizes in batch
        s2_weight_maps = {}

        if self.use_stage2 and depth_maps is not None and water_types is not None:
            _depth_maps   = depth_maps
            _water_types  = water_types
            _ppcm_s2      = self.ppcm_s2

            def fpn_hook(module, input, output):
                import torch.nn.functional as F
                from iop_table import DEPTH_MIN, DEPTH_MAX, get_dominant_beta

                modulated = {}

                for key, feat in output.items():
                    B, C, H, W   = feat.shape
                    weighted_list = []

                    for b in range(B):
                        single_feat  = feat[b:b+1]           # (1, C, H, W)
                        single_depth = _depth_maps[b]        # (1, H_orig, W_orig)
                        single_wt    = _water_types[b]

                        beta = get_dominant_beta(single_wt)
                        z    = DEPTH_MIN + single_depth * (DEPTH_MAX - DEPTH_MIN)

                        # Resize depth to this FPN level resolution
                        z_r = F.interpolate(
                            z.unsqueeze(0),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        )  # (1, 1, H, W)

                        weight = torch.exp(
                            -beta * _ppcm_s2.depth_scale * z_r
                        )
                        weight = weight / (weight.mean() + 1e-8)
                        weighted_list.append(single_feat * weight)

                        # Save first image's weight map for visualization
                        if b == 0 and key not in s2_weight_maps:
                            s2_weight_maps[key] = weight.detach()

                    modulated[key] = torch.cat(weighted_list, dim=0)

                return modulated

            hook = self.detector.backbone.register_forward_hook(fpn_hook)
        else:
            hook = None

        # ── Run FasterRCNN ─────────────────────────────────────────────
        if self.training:
            losses = self.detector(images, targets)

            if hook:
                hook.remove()

            if s2_weight_maps:
                self.last_viz['stage2'] = {
                    k: v.cpu() for k, v in s2_weight_maps.items()
                }

            return losses

        else:
            detections = self.detector(images)

            if hook:
                hook.remove()

            if s2_weight_maps:
                self.last_viz['stage2'] = {
                    k: v.cpu() for k, v in s2_weight_maps.items()
                }

            return detections

    def get_stage1_alpha(self):
        return self.ppcm_s1.alpha if self.ppcm_s1 else 0.0

    def get_stage2_depth_scale(self):
        return self.ppcm_s2.depth_scale if self.ppcm_s2 else 0.0
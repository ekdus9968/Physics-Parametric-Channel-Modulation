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

        # ── PPCM Stage 1: pixel-space correction ──────────────────────
        if self.use_stage1 and depth_maps is not None and water_types is not None:
            corrected_images  = []
            s1_viz_raw        = []
            s1_viz_corrected  = []
            s1_viz_backscatter = []

            for img, depth, wt in zip(images, depth_maps, water_types):
                raw_4d   = img.unsqueeze(0)         # (1,3,H,W)
                depth_4d = depth.unsqueeze(0)       # (1,1,H,W)

                corrected, bscat = self.ppcm_s1(raw_4d, wt, depth_4d)

                corrected_images.append(corrected.squeeze(0))
                s1_viz_raw.append(img.detach().cpu())
                s1_viz_corrected.append(corrected.squeeze(0).detach().cpu())
                s1_viz_backscatter.append(bscat.squeeze(0).detach().cpu())

            images = corrected_images  # pass corrected to backbone

            self.last_viz['stage1'] = {
                'raw':        s1_viz_raw,
                'corrected':  s1_viz_corrected,
                'backscatter': s1_viz_backscatter,
            }
        else:
            corrected_images = images

        # ── PPCM Stage 2: FPN spatial weighting ───────────────────────
        # Hook on backbone to intercept FPN output
        s2_weight_maps = {}

        if self.use_stage2 and depth_maps is not None and water_types is not None:
            # Use first image's water type for batch
            # (per-image would need per-image FPN processing)
            batch_wt    = water_types[0]
            batch_depth = torch.stack(depth_maps, dim=0)  # (B,1,H,W)

            def fpn_hook(module, input, output):
                modulated, wmaps = self.ppcm_s2(output, batch_wt, batch_depth)
                s2_weight_maps.update(wmaps)
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
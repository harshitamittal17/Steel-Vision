"""
=================================================================================
 MicroDefect-AI: Steel Surface Micro-Crack & Defect Recognition Engine
 Model Architecture: EfficientNet-B0 (Transfer Learning) + Multi-Scale Head
 Explainable AI: Grad-CAM with Morphological Crack Contour Extraction
=================================================================================
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import numpy as np

# 6 NEU Surface Defect Database categories
DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

METALLURGICAL_SPECS = {
    "crazing": {
        "hazard_rating": "Critical",
        "description": "Network of fine thermal/stress fatigue micro-cracks that propagate rapidly under cyclic tensile load.",
        "tolerance": "Zero tolerance (Reject)",
    },
    "inclusion": {
        "hazard_rating": "High",
        "description": "Non-metallic particulate traps (slag/oxides) weakening structural yield strength.",
        "tolerance": "Maximum 2mm isolated",
    },
    "patches": {
        "hazard_rating": "Moderate",
        "description": "Localized surface irregularities and thickness variances caused by uneven reduction rolls.",
        "tolerance": "Allowable if within roughness Ra < 3.2 um",
    },
    "pitted_surface": {
        "hazard_rating": "Critical",
        "description": "Localized micro-cavities induced by acid pickling or chemical pitting; major stress-riser sites.",
        "tolerance": "Zero tolerance (Reject)",
    },
    "rolled-in_scale": {
        "hazard_rating": "High",
        "description": "Dense iron oxide layer pressed mechanically into the steel sheet during hot strip rolling.",
        "tolerance": "Requires abrasive de-scaling",
    },
    "scratches": {
        "hazard_rating": "Low",
        "description": "Linear abrasive scoring caused by guide friction or tooling contact; typically superficial.",
        "tolerance": "Acceptable up to grade standard 1b",
    }
}


class MicroCrackEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 configured with a specialized classification head
    designed for high sensitivity towards fine micro-cracks and surface flaws.
    """
    def __init__(self, num_classes=6, pretrained=False):
        super().__init__()
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except Exception:
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_channels = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_channels, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self):
        # Final convolutional feature map layer in EfficientNet-B0
        return self.backbone.features[-1]


class MicroGradCAM:
    """
    Grad-CAM implementation with gradient backpropagation and 
    morphological micro-crack geometric profiling.
    """
    def __init__(self, model):
        self.model = model
        self.target_layer = model.get_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def f_hook(mod, inp, out):
            self.activations = out.detach()

        def b_hook(mod, g_in, g_out):
            self.gradients = g_out[0].detach()

        self.target_layer.register_forward_hook(f_hook)
        self.target_layer.register_full_backward_hook(b_hook)

    def compute_cam(self, tensor_img, class_idx=None):
        self.model.eval()
        x = tensor_img.clone().requires_grad_(True)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = x.to(device)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        grads = self.gradients[0]     # [C, H, W]
        acts = self.activations[0]    # [C, H, W]

        # Global average pooling of gradients
        weights = grads.mean(dim=(1, 2))  # [C]
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0)).cpu().numpy()

        # Min-max normalization
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam, class_idx


def extract_crack_morphology(cam_map, raw_rgb_224, cam_threshold=0.45):
    """
    Extracts geometric contour profiles and crack features directly from the Grad-CAM activation.
    Returns:
      - Binary crack mask
      - Outlined visual image with contours & bounding boxes
      - Geometric statistics (major length, minor width, orientation, affected area)
    """
    binary_mask = (cam_map >= cam_threshold).astype(np.uint8) * 255
    
    # Morphological opening to clean small speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = raw_rgb_224.copy()
    contours_found = []
    total_area_px = 0

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 15:  # Ignore tiny noise fragments
            continue

        total_area_px += area
        # Draw contour boundary in bright cyan
        cv2.drawContours(annotated, [cnt], -1, (0, 240, 255), 2)

        # Fit rotated bounding rectangle to measure micro-crack dimensions
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (axis1, axis2), angle = ellipse
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            cv2.ellipse(annotated, ellipse, (255, 60, 0), 1)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            major_axis = max(w, h)
            minor_axis = min(w, h)
            angle = 0.0

        contours_found.append({
            "id": idx + 1,
            "area_px": round(area, 1),
            "major_length_px": round(major_axis, 1),
            "aspect_ratio": round(major_axis / max(minor_axis, 1.0), 2),
            "angle_deg": round(angle, 1)
        })

    surface_coverage_pct = (total_area_px / (224 * 224)) * 100.0

    return {
        "mask": cleaned_mask,
        "annotated_image": annotated,
        "features": contours_found,
        "crack_count": len(contours_found),
        "surface_coverage_pct": round(surface_coverage_pct, 2)
    }

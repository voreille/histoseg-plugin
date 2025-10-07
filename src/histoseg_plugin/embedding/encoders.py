# src/histoseg_plugin/embedding/encoders.py
from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn
from torchvision import transforms as T

# add more as needed
_VALID = {"resnet50_trunc", "uni_v1", "conch_v1"}


def get_encoder(name: str,
                target_img_size: int = 224) -> Tuple[nn.Module, T.Compose]:
    name = name.lower()
    if name == "resnet50_trunc":
        m = timm.create_model("resnet50", pretrained=True,
                              num_classes=0)  # global pool features
        tx = T.Compose([
            T.Resize(target_img_size),
            T.CenterCrop(target_img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return m, tx
    elif name == "uni_v1":
        # TODO: replace with your encoder + proper transforms
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        tx = T.Compose([
            T.Resize(target_img_size),
            T.CenterCrop(target_img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        return m, tx
    elif name == "conch_v1":
        # TODO
        m = timm.create_model("efficientnet_b0",
                              pretrained=True,
                              num_classes=0)
        tx = T.Compose([
            T.Resize(target_img_size),
            T.CenterCrop(target_img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return m, tx
    else:
        raise ValueError(f"Unknown encoder '{name}'. Valid: {_VALID}")

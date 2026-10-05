"""Frozen DINOv3 image encoder producing patch tokens."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

DEFAULT_ENCODER = "vit_large_patch16_dinov3.lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoPatchEncoder(nn.Module):
    """Wraps a timm DINOv3 model and returns only its patch tokens.

    Inputs must be ImageNet-normalised ``(B, 3, H, W)`` tensors. Class and register tokens
    are removed so the output is ``(B, (H / patch) * (W / patch), embed_dim)``. The encoder
    is frozen and always run in evaluation mode.
    """

    def __init__(self, model_name: str = DEFAULT_ENCODER, *, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        self.embed_dim = int(self.model.embed_dim)
        self.patch_size = int(self.model.patch_embed.patch_size[0])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # noqa: D401 - keep the encoder frozen in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)
        return features[:, self.num_prefix_tokens :, :]


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Map ``[0, 1]`` float images ``(..., 3, H, W)`` to ImageNet-normalised inputs."""
    mean = images.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (images - mean) / std

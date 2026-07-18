"""Attention blocks used by the improved YOLOv8 architecture."""

from __future__ import annotations

import torch
from torch import nn


class SEResearch(nn.Module):
    """Squeeze-and-Excitation channel attention (eager, shape-preserving).

    A single lightweight attention block used for the +SE ablation. Built entirely
    in __init__ (no lazy modules) so every parameter exists before the optimizer is
    created and is therefore included in training.

    Identity init: the last conv is zero-initialised with a positive bias so the gate
    starts near 1 (~0.95). When inserted into a PRETRAINED network this lets the
    pretrained features pass through almost unchanged at the start, and SE learns to
    modulate them gradually — instead of a random sigmoid≈0.5 that halves the features
    and destroys the pretrained head.
    """

    def __init__(self, channels: int, reduction: int = 16, init_bias: float = 3.0) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

        # Start near-identity: gate ≈ sigmoid(init_bias) ≈ 0.95
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class ChannelAttention(nn.Module):
    """CBAM channel attention using shared MLP weights."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """CBAM spatial attention from channel-wise average and maximum maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM spatial kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return self.gate(self.conv(torch.cat((avg_map, max_map), dim=1)))


class CBAMResearch(nn.Module):
    """Convolutional Block Attention Module with unchanged tensor shape."""

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)

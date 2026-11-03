"""Hand-written U-Net (Ronneberger et al., 2015) without a pretrained encoder.

Returns raw logits; apply sigmoid only for metrics / inference.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2. Padding keeps the spatial size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class VanillaUNet(nn.Module):
    """Classic U-Net.

    Args:
        in_channels: number of input image channels.
        classes: number of output channels (1 for binary segmentation).
        base_channels: channels at the first level; doubled at every downsampling.
        depth: number of downsampling steps (the bottleneck is at resolution H / 2**depth).
    """

    def __init__(self, in_channels: int = 3, classes: int = 1, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.encoder = nn.ModuleList()
        prev = in_channels
        for ch in channels[:-1]:
            self.encoder.append(DoubleConv(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for ch_in, ch_out in zip(reversed(channels[1:]), reversed(channels[:-1]), strict=True):
            self.upsample.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2))
            # After concatenation with the skip connection: ch_out (up) + ch_out (skip)
            self.decoder.append(DoubleConv(2 * ch_out, ch_out))

        self.head = nn.Conv2d(channels[0], classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.upsample, self.decoder, reversed(skips), strict=True):
            x = up(x)
            # Input sizes not divisible by 2**depth lose a pixel when pooling; realign to the skip
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))

        return self.head(x)

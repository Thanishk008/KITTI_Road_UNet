from __future__ import annotations

from typing import Dict, Type

import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int, requested_groups: int = 8) -> nn.GroupNorm:
    groups = min(requested_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            group_norm(out_channels, groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            group_norm(out_channels, groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = group_norm(out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = group_norm(out_channels, groups)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: Type[nn.Module], groups: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), block(in_channels, out_channels, groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, block: Type[nn.Module], groups: int, use_skip: bool):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        conv_in = out_channels + (skip_channels if use_skip else 0)
        self.conv = block(conv_in, out_channels, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_skip:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBase(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        depth: int = 4,
        groups: int = 8,
        block: Type[nn.Module] = ConvBlock,
        use_skip: bool = True,
    ):
        super().__init__()
        if depth != 4:
            raise ValueError("This project fixes depth=4 for a clear report computation graph.")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.inc = block(3, channels[0], groups)
        self.down1 = Down(channels[0], channels[1], block, groups)
        self.down2 = Down(channels[1], channels[2], block, groups)
        self.down3 = Down(channels[2], channels[3], block, groups)
        self.down4 = Down(channels[3], channels[4], block, groups)
        self.up1 = Up(channels[4], channels[3], channels[3], block, groups, use_skip)
        self.up2 = Up(channels[3], channels[2], channels[2], block, groups, use_skip)
        self.up3 = Up(channels[2], channels[1], channels[1], block, groups, use_skip)
        self.up4 = Up(channels[1], channels[0], channels[0], block, groups, use_skip)
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.head(x)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits


class PlainUNet(UNetBase):
    def __init__(self, base_channels: int = 32, depth: int = 4, group_norm_groups: int = 8):
        super().__init__(base_channels, depth, group_norm_groups, ConvBlock, use_skip=True)


class NoSkipUNet(UNetBase):
    def __init__(self, base_channels: int = 32, depth: int = 4, group_norm_groups: int = 8):
        super().__init__(base_channels, depth, group_norm_groups, ConvBlock, use_skip=False)


class ResidualRoadUNet(UNetBase):
    def __init__(self, base_channels: int = 32, depth: int = 4, group_norm_groups: int = 8):
        super().__init__(base_channels, depth, group_norm_groups, ResidualBlock, use_skip=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class PretrainedResNet34UNet(nn.Module):
    """U-Net-style decoder on a torchvision ResNet-34 encoder.

    This is an advanced transfer-learning comparison. The course-designed
    from-scratch U-Net models remain implemented above.
    """

    def __init__(self, base_channels: int = 64, depth: int = 4, group_norm_groups: int = 8, pretrained: bool = True):
        super().__init__()
        if depth != 4:
            raise ValueError("ResNet34 U-Net uses four encoder stages.")
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("torchvision is required for resnet34_unet_pretrained. Install requirements.txt.") from exc

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        decoder_channels = max(16, int(base_channels))
        self.up4 = DecoderBlock(512, 256, decoder_channels * 8, group_norm_groups)
        self.up3 = DecoderBlock(decoder_channels * 8, 128, decoder_channels * 4, group_norm_groups)
        self.up2 = DecoderBlock(decoder_channels * 4, 64, decoder_channels * 2, group_norm_groups)
        self.up1 = DecoderBlock(decoder_channels * 2, 64, decoder_channels, group_norm_groups)
        self.head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        stem = self.stem(x)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, stem)
        logits = self.head(x)
        return F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)


MODEL_REGISTRY = {
    "residual_unet": ResidualRoadUNet,
    "plain_unet": PlainUNet,
    "no_skip_unet": NoSkipUNet,
    "resnet34_unet_pretrained": PretrainedResNet34UNet,
}


def build_model(config: Dict) -> nn.Module:
    name = config.get("name", "residual_unet")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    kwargs = {
        "base_channels": int(config.get("base_channels", 32)),
        "depth": int(config.get("depth", 4)),
        "group_norm_groups": int(config.get("group_norm_groups", 8)),
    }
    if name == "resnet34_unet_pretrained":
        kwargs["pretrained"] = bool(config.get("pretrained", True))
    return MODEL_REGISTRY[name](**kwargs)

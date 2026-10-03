# original version: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# this is specifically for Qwen Image 2.1: x -> (b c t h w) where t is always 1


from diffusers.configuration_utils import ConfigMixin, register_to_config

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vae_wan import AttentionBlock, RMS_norm
from .vae_wan22 import AvgDown3D, DupUp3D


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = nn.Conv3d(dim, dim * 2, (1, 1, 1))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = nn.Conv3d(dim, dim, (1, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = rearrange(x, "b c t h w -> (b t) c h w")
        if self.mode in ("upsample2d", "upsample3d"):
            x = strip_apply(self.resample, x, scale=2)
        else:
            x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=1)

        return x


STRIP_ELEMS = 2 ** 24


def strip_apply(fn, x, scale=1, halo=1, out=None):
    # strips of rows bound cudnn's conv workspace, a halo row per 3x3 conv keeps them exact
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x)
    add = out is not None
    size = x.shape[-2]
    step = -(-size // n)
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo)).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        dst = out.narrow(-2, a * scale, (b - a) * scale)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (1, 3, 3), padding=(0, 1, 1)),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, out_dim, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.residual[2]._padding = 0
        self.residual[6]._padding = 0
        
        self.shortcut = (
            nn.Conv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        # single image: the whole block runs in strips so its intermediates never exist at full size
        return strip_apply(lambda s: self.residual(s).add_(self.shortcut(s)), x, halo=2)


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x):
        x_copy = x
        for module in self.downsamples:
            x = module(x)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_upsample=False, up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x):
        x_main = x
        for module in self.upsamples:
            x_main = module(x_main)
        if self.avg_shortcut is not None:
            return strip_apply(lambda s: self.avg_shortcut(s), x, scale=self.avg_shortcut.factor_s, halo=0, out=x_main)
        else:
            return x_main


class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, temporal_downsample=[True, True, False], dropout=0.0):
        super().__init__()

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = nn.Conv3d(4, dims[0], (1, 3, 3), padding=(0, 1, 1))
        self.conv1._padding = 0

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temporal_downsample[i]
                if i < len(temporal_downsample) else False)
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                ))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Conv3d(out_dim, z_dim, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.head[2]._padding = 0

    def forward(self, x):
        x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x)

        for layer in self.middle:
            x = layer(x)

        for layer in self.head:
            x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, temporal_upsample=[False, True, True], dropout=0.0):
        super().__init__()

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # init block
        self.conv1 = nn.Conv3d(z_dim, dims[0], (1, 3, 3), padding=(0, 1, 1))
        self.conv1._padding = 0

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temporal_upsample[i] if i < len(
                temporal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temporal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                ))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Conv3d(out_dim, 4, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.head[2]._padding = 0

    def forward(self, x):
        x = self.conv1(x)

        for layer in self.middle:
            x = layer(x)

        for layer in self.upsamples:
            x = layer(x)

        for layer in self.head:
            x = layer(x)

        return x


class AutoencoderQwen21(nn.Module, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        base_dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_downsample=[True, True, False], # False, True, True, True in config
        dropout=0.0,
        image_channels=3,
        patch_size=1,
    ):
        super().__init__()
        temporal_upsample = temporal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            base_dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            temporal_downsample,
            dropout,
        )
        self.conv1 = nn.Conv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            temporal_upsample,
            dropout,
        )
        self.latents_mean = torch.tensor([
            0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
            1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
            2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
            -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
            -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
            -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
            -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
            0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
        ]).view(1, 64, 1, 1)
        self.latents_std = torch.tensor([
            3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
            3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
            3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
            3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
            3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
            3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
            3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
            3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
        ]).view(1, 64, 1, 1)

    def encode(self, x): # x: BCHW
        if x.shape[1] == 3: # expects four channels
            mask = x.new_ones((x.shape[0], 1, x.shape[2], x.shape[3]))
            x = torch.cat((x, mask), dim=1)
        return self.conv1(self.encoder(x.unsqueeze(2))).chunk(2, dim=1)[0].squeeze(2)

    def decode(self, z):
        out = self.decoder(self.conv2(z.unsqueeze(2)))
        return out.squeeze(2)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std + latents_mean

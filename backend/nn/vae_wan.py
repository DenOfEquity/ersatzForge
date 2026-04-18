# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# this is cut-down for single frame generation, by DoE

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from backend.attention import attention_function_single_head_spatial
from diffusers.configuration_utils import ConfigMixin, register_to_config


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        padding = list(self._padding)
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'downsample2d')
        super().__init__()

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        if x.ndim == 4:
            x = self.resample(x.to(torch.float32)).type_as(x)
        else: # x.ndim == 5
            b, c, t, h, w = x.size()
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.resample(x.to(torch.float32)).type_as(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.optimized_attention = attention_function_single_head_spatial

    def forward(self, x):
        identity = x.clone()
        if x.ndim == 5:
             x = rearrange(x, 'b c t h w -> (b t) c h w')
       
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = self.optimized_attention(q, k, v)
        x = self.proj(x)

        if identity.ndim == 5:
            x = rearrange(x, '(b t) c h w-> b c t h w', t=1)

        return x + identity


class Encoder3d(nn.Module):
    def __init__(self,
                 dim=96,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, mode='downsample2d'))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

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
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, mode='upsample2d'))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)

        for layer in self.middle:
            x = layer(x)

        for layer in self.upsamples:
            x = layer(x)

        for layer in self.head:
            x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class AutoencoderKLWan(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temporal_downsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, dropout)

        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, 16, 1, 1)
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, 16, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean

    def encode(self, x):
        x = x.unsqueeze(2)
        
        out = self.encoder(x)
        mu, log_var = self.conv1(out).chunk(2, dim=1)

        return mu.squeeze(2)

    def decode(self, z):
        z = z.unsqueeze(2)    # z: [b,c,t,h,w]

        x = self.conv2(z)
        out = self.decoder(x)
        return out.squeeze(2)


### Anzhc's 2D WAN VAE


class QwenImageResidualBlock2D(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.norm1 = RMS_norm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h


class QwenImageMidBlock2D(nn.Module):
    def __init__(self, dim, dropout=0.0, num_layers=1):
        super().__init__()
        resnets = [QwenImageResidualBlock2D(dim, dim, dropout)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock(dim))
            resnets.append(QwenImageResidualBlock2D(dim, dim, dropout))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


class QwenImageEncoder2D(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=32,
        input_channels=3,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        dropout=0.0,
    ):
        super().__init__()
        self.dim_mult = list(dim_mult)
        self.attn_scales = list(attn_scales)

        dims = [dim * multiplier for multiplier in [1] + self.dim_mult]
        scale = 1.0

        self.conv_in = nn.Conv2d(input_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock2D(in_dim, out_dim, dropout))
                if scale in self.attn_scales:
                    self.down_blocks.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if index != len(self.dim_mult) - 1:
                self.down_blocks.append(Resample(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock2D(out_dim, dropout, num_layers=1)
        self.norm_out = RMS_norm(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class QwenImageUpBlock2D(nn.Module):
    def __init__(self, in_dim, out_dim, num_res_blocks, dropout=0.0, upsample_mode=None):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock2D(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([Resample(out_dim, mode=upsample_mode)])

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageDecoder2D(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        output_channels=3,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        dropout=0.0,
    ):
        super().__init__()
        self.dim_mult = list(dim_mult)

        dims = [dim * multiplier for multiplier in [self.dim_mult[-1]] + self.dim_mult[::-1]]

        self.conv_in = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock2D(dims[0], dropout, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if index > 0:
                in_dim = in_dim // 2
            upsample_mode = "upsample2d" if index != len(self.dim_mult) - 1 else None
            self.up_blocks.append(
                QwenImageUpBlock2D(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                )
            )

        self.norm_out = RMS_norm(out_dim)
        self.conv_out = nn.Conv2d(out_dim, output_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class AutoencoderQwen2D(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(
        self,
        base_dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temporal_downsample=(False, True, True),
        image_channels=3,
        dropout=0.0,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temporal_downsample = list(temporal_downsample)
        self.temporal_upsample = list(self.temporal_downsample[::-1])

        self.encoder = QwenImageEncoder2D(
            dim=base_dim,
            z_dim=z_dim * 2,
            input_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            dropout=dropout,
        )
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder2D(
            dim=base_dim,
            z_dim=z_dim,
            output_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            dropout=dropout,
        )

        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, 16, 1, 1)
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, 16, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean

    def encode(self, x, **kwargs):
        moments = self.quant_conv(self.encoder(x))
        mu, _ = moments.chunk(2, dim=1)
        return mu

    def decode(self, z, **kwargs):
        out = self.decoder(self.post_quant_conv(z))
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out

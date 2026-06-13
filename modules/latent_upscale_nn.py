import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


#### #### #### ####

# City96's latent upscalers for sd1, sd2, sdxl
# based on https://github.com/city96/SD-Latent-Upscaler/

class Upscaler(nn.Module):
    """
        Basic NN layout, ported from:
        https://github.com/city96/SD-Latent-Upscaler/blob/main/upscaler.py
    """
    version = 2.1 # network revision
    def head(self):
        return [
            nn.Conv2d(self.chan, self.size, kernel_size=self.krn, padding=self.pad),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.fac, mode="nearest"),
            nn.ReLU(),
        ]
    def core(self):
        layers = []
        for _ in range(self.depth):
            layers += [
                nn.Conv2d(self.size, self.size, kernel_size=self.krn, padding=self.pad),
                nn.ReLU(),
            ]
        return layers
    def tail(self):
        return [
            nn.Conv2d(self.size, self.chan, kernel_size=self.krn, padding=self.pad),
        ]

    def __init__(self, fac, depth=16):
        super().__init__()
        self.size = 64      # Conv2d size
        self.chan = 4       # in/out channels
        self.depth = depth  # no. of layers
        self.fac = fac      # scale factor
        self.krn = 3        # kernel size
        self.pad = 1        # padding

        self.sequential = nn.Sequential(
            *self.head(),
            *self.core(),
            *self.tail(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


def upscale(samples, latent_ver, scale_factor):
    model = Upscaler(scale_factor)
    filename = f"latent-upscaler-v{model.version}_SD{latent_ver}-x{scale_factor}.safetensors"

    weights = str(hf_hub_download(
        repo_id="city96/SD-Latent-Upscaler",
        filename=filename,
        revision="99c65021fa947dfe3d71ec4e24793fe7533a3322",    # specifying revision avoids check
        )
    )

    model.load_state_dict(load_file(weights))
    lt = model(samples.to(torch.float32).cpu())
    del model

    return lt.to(samples.device, dtype=samples.dtype)


#### #### #### ####

# TensorForger's flow upscaler for flux2 latents
# based on https://github.com/tensorforger/comfyui-flow-upscaler

from diffusers import FlowMatchEulerDiscreteScheduler

from modules import devices


def make_group_norm(channels: int, max_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=eps)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2

        freqs = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=timesteps.device))
            * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype)
            / half
        )
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))

        return emb


class ConditioningEncoder(nn.Module):
    def __init__(self, time_dim: int = 128, cond_dim: int = 256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_vec = self.time_proj(self.time_embed(timestep))
        return time_vec


class ConditionedResidualBlock(nn.Module):
    """
    SDXL-style residual block:
      GN -> SiLU -> Conv
      + condition (scale/shift)
      GN -> SiLU -> Dropout -> Conv
      + skip connection
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = make_group_norm(input_channels)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * output_channels),
        )

        self.norm2 = make_group_norm(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )

        if input_channels != output_channels:
            self.skip = nn.Conv2d(
                input_channels, output_channels, kernel_size=1, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = nn.functional.silu(h)
        h = self.conv1(h)

        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class DownStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        cond_dim: int = 256,
        dropout: float = 0.0,
        num_blocks: int = 1,
        downsample_first: bool = False,
    ):
        super().__init__()
        self.downsample_first = downsample_first

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = input_channels if i == 0 else output_channels
            self.blocks.append(
                ConditionedResidualBlock(
                    input_channels=in_ch,
                    output_channels=output_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )

        self.downsample = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        if self.downsample_first:
            x = self.downsample(x)

        for block in self.blocks:
            x = block(x, cond)
        skip = x

        if not self.downsample_first:
            x = self.downsample(x)

        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        cond_dim: int = 256,
        dropout: float = 0.0,
        num_blocks: int = 1,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = (input_channels + skip_channels) if i == 0 else output_channels
            self.blocks.append(
                ConditionedResidualBlock(
                    input_channels=in_ch,
                    output_channels=output_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)

        for block in self.blocks:
            x = block(x, cond)

        return x


class LowResEncoder(nn.Module):
    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 128,
        cond_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_conv = nn.Conv2d(
            sample_channels, base_channels, kernel_size=1, padding=0
        )

        self.block_1 = ConditionedResidualBlock(
            input_channels=base_channels,
            output_channels=base_channels,
            cond_dim=cond_dim,
            dropout=dropout,
        )

        self.block_2 = DownStage(
            input_channels=base_channels,
            output_channels=base_channels,
            cond_dim=cond_dim,
            dropout=dropout,
            num_blocks=1,
            downsample_first=True,
        )

        self.block_3 = DownStage(
            input_channels=base_channels,
            output_channels=base_channels,
            cond_dim=cond_dim,
            dropout=dropout,
            num_blocks=1,
            downsample_first=True,
        )

    def forward(self, latents_small, cond):
        x = self.in_conv(latents_small)
        block_1_out = self.block_1(x, cond)
        block_2_out, _ = self.block_2(block_1_out, cond)
        block_3_out, _ = self.block_3(block_2_out, cond)

        return block_1_out, block_2_out, block_3_out


class FilmCond2D(nn.Module):
    def __init__(self, base_channels: int = 256, cond_channels: int = 256):
        super().__init__()

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(cond_channels, base_channels * 2, kernel_size=1),
        )

    def forward(self, x, cond):
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)

        x = x * (1 + scale) + shift

        return x


class Flux2_UpscalerUNet(nn.Module):
    def __init__(
        self,
        sample_channels: int = 32,
        base_channels: int = 384,
        time_dim: int = 512,
        cond_dim: int = 1024,
        dropout: float = 0.01,
    ):
        super().__init__()

        self.conditioning = ConditioningEncoder(
            time_dim=time_dim,
            cond_dim=cond_dim,
        )

        self.in_conv = nn.Conv2d(sample_channels, base_channels, kernel_size=1, padding=0)

        self.low_res_encoder = LowResEncoder(base_channels=base_channels)

        self.film_cond_1 = FilmCond2D(base_channels=base_channels, cond_channels=base_channels)
        self.film_cond_2 = FilmCond2D(base_channels=base_channels, cond_channels=base_channels)
        self.film_cond_3 = FilmCond2D(base_channels=base_channels, cond_channels=base_channels)

        self.down_stages = nn.ModuleList(
            [
                DownStage(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=3,
                ),
                DownStage(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=2,
                ),
            ]
        )

        self.mid_stages = nn.ModuleList(
            [
                ConditionedResidualBlock(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
                for i in range(1)
            ]
        )

        self.up_stages = nn.ModuleList(
            [
                UpStage(
                    input_channels=base_channels,
                    skip_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=2,
                ),
                UpStage(
                    input_channels=base_channels,
                    skip_channels=base_channels,
                    output_channels=base_channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    num_blocks=3,
                ),
            ]
        )

        self.out_conv = nn.Conv2d(
            base_channels, sample_channels, kernel_size=1, padding=0
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, latents_small: torch.Tensor
    ) -> torch.Tensor:
        cond = self.conditioning(timestep)

        B, C, H, W = sample.shape

        lr_cond_1, lr_cond_2, lr_cond_3 = self.low_res_encoder(latents_small, cond)

        lr_cond_1 = torch.nn.functional.interpolate(lr_cond_1, (H, W), mode="bilinear")
        lr_cond_2 = torch.nn.functional.interpolate(lr_cond_2, (H // 2, W // 2), mode="bilinear")
        lr_cond_3 = torch.nn.functional.interpolate(lr_cond_3, (H // 4, W // 4), mode="bilinear")

        x = self.in_conv(sample)
        x = self.film_cond_1(x, lr_cond_1)

        skips = []

        x, skip = self.down_stages[0](x, cond)
        skips.append(skip)

        x = self.film_cond_2(x, lr_cond_2)

        x, skip = self.down_stages[1](x, cond)
        skips.append(skip)

        x = self.film_cond_3(x, lr_cond_3)

        for mid in self.mid_stages:
            x = mid(x, cond)

        for up in self.up_stages:
            x = up(x, skips.pop(), cond)

        x = self.out_conv(x)
        return x


def upscale_f2(samples):
    flow_upscaler = Flux2_UpscalerUNet()

    weights = str(hf_hub_download(
        repo_id="TensorForger/FlowUpscaler",
        filename="flow_upscaler.safetensors",
        revision="bfc01abb0a8c8bfa4861feaf003d64da17154d46",    # specifying revision avoids check
        )
    )

    state_dict = load_file(weights)
    flow_upscaler.load_state_dict(state_dict)
    flow_upscaler.eval()

    device = devices.device

    latents_small = samples.to(device)

    batch_size = latents_small.shape[0]
    target_latent_height = latents_small.shape[2] * 2
    target_latent_width = latents_small.shape[3] * 2

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    scheduler = FlowMatchEulerDiscreteScheduler()

    scheduler.set_timesteps(1, mu=1.0)
    latents = torch.normal(
        mean=0,
        std=1,
        size=(batch_size, 32, target_latent_height, target_latent_width),
        dtype=latents_small.dtype,
        device=device,
        generator=generator,
    )

    flow_upscaler = flow_upscaler.to(device)

    for t in scheduler.timesteps:
        latent_model_input = latents
        t = t.to(device).view(1)
        predicted_noise = flow_upscaler(
            sample=latent_model_input,
            timestep=t,
            latents_small=latents_small,
        )
        latents = scheduler.step(predicted_noise, t, latents).prev_sample

    del flow_upscaler

    return latents.to(samples.device, dtype=samples.dtype)

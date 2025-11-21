# based on https://github.com/livinginparadise/GRDDenoiser/ and using model trained by livinginparadise

from modules import scripts_postprocessing, ui_components
from modules.paths_internal import models_path
import gradio as gr

import torch
import numpy
from safetensors.torch import load_file as load_safetensors
import os
from PIL import Image

from typing import Any, Dict, List, Sequence, Tuple


QUALITY_GEOM_TRANSFORMS: List[Tuple[int, bool]] = [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
]
QUALITY_GAIN_FACTORS: Tuple[float, ...] = (0.95, 1.0, 1.05)


class BiasFreeConv(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bias", False)
        super().__init__(*args, **kwargs)


class LayerNorm2d(torch.nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        hidden = channels * expansion
        self.depth = BiasFreeConv(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
        )
        self.point = BiasFreeConv(channels, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point(self.depth(x))


class GlobalGating(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = BiasFreeConv(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x.mean(dim=(2, 3), keepdim=True)
        gate = torch.sigmoid(self.proj(gate))
        return gate


class GatedResidualBlock(torch.nn.Module):
    def __init__(self, channels: int, expansion: int = 2, kernel_size: int = 5):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.local = DepthwiseSeparableConv(
            channels, expansion=expansion, kernel_size=kernel_size
        )
        self.activation = torch.nn.GELU()
        self.proj = BiasFreeConv(channels * expansion, channels, kernel_size=1)
        self.gate = GlobalGating(channels)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        if cond is not None:
            gamma, beta = cond
            out = out * (1.0 + gamma) + beta
        out = self.local(out)
        out = self.activation(out)
        out = self.proj(out)
        out = out * self.gate(residual)
        return residual + out


class ResidualStage(torch.nn.Module):
    def __init__(self, channels: int, num_blocks: int, expansion: int = 2):
        super().__init__()
        self.channels = channels
        self.blocks = torch.nn.ModuleList(
            GatedResidualBlock(channels, expansion=expansion) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        block_cond = None
        if cond is not None:
            if cond.dim() == 2:
                cond = cond.unsqueeze(-1).unsqueeze(-1)
            gamma, beta = torch.chunk(cond, 2, dim=1)
            gamma = gamma[:, : self.channels]
            beta = beta[:, : self.channels]
            block_cond = (gamma, beta)
        for block in self.blocks:
            x = block(x, block_cond)
        return x


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


class ResidualPriorExtractor(torch.nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int]):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        for idx, k in enumerate(self.kernel_sizes):
            sigma = max(0.8, float(k) / 3.0)
            kernel = _gaussian_kernel(k, sigma).view(1, 1, k, k)
            self.register_buffer(f"kernel_{idx}", kernel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        priors: List[torch.Tensor] = []
        descriptors: List[torch.Tensor] = []
        b, c, _, _ = x.shape
        for idx, k in enumerate(self.kernel_sizes):
            kernel = getattr(self, f"kernel_{idx}").to(dtype=x.dtype, device=x.device)
            kernel = kernel.expand(c, 1, k, k)
            smooth = torch.nn.functional.conv2d(x, kernel, padding=k // 2, groups=c)
            residual = x - smooth
            priors.append(residual)
            descriptors.append(residual.abs().mean(dim=(1, 2, 3), keepdim=True))
        prior_map = torch.cat(priors, dim=1)
        descriptor = torch.cat(descriptors, dim=1).view(b, -1)
        return prior_map, descriptor


class NoiseConditioner(torch.nn.Module):
    def __init__(
        self,
        stage_channels: Sequence[int],
        hidden_dim: int = 96,
        descriptor_dim: int = 0,
    ):
        super().__init__()
        self.stage_channels = list(stage_channels)
        in_dim = 1 + max(0, descriptor_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.heads = torch.nn.ModuleList(
            torch.nn.Linear(hidden_dim, 2 * channels) for channels in self.stage_channels
        )

    def forward(
        self, sigma: torch.Tensor, descriptor: torch.Tensor | None
    ) -> List[torch.Tensor]:
        sigma = sigma.view(sigma.shape[0], 1)
        sigma = sigma.clamp(min=1e-5)
        embedding = torch.log(sigma * 255.0)
        if descriptor is not None:
            embedding = torch.cat([embedding, descriptor], dim=1)
        base = self.encoder(embedding)
        conds = [head(base).unsqueeze(-1).unsqueeze(-1) for head in self.heads]
        return conds


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    tensor = t.clamp(0.0, 1.0).detach().cpu()
    arr = tensor.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(numpy.uint8)
    return Image.fromarray(arr[:,:,::-1])


class BiasFreeResidualDenoiser(torch.nn.Module):
    def __init__(
        self,
        image_channels: int,
        base_channels: int = 48,
        depth: int = 3,
        blocks_per_stage: int = 2,
        residual_kernel_sizes: Sequence[int] = (3, 5, 7),
    ):
        super().__init__()
        self.image_channels = image_channels
        self.prior_extractor = ResidualPriorExtractor(
            image_channels, kernel_sizes=residual_kernel_sizes
        )
        self.prior_scales = len(tuple(residual_kernel_sizes))
        self.prior_channels = self.prior_scales * image_channels
        self.input_channels = image_channels + 1 + self.prior_channels

        encoder_channels = [base_channels * (2**i) for i in range(depth)]
        self.entry = BiasFreeConv(self.input_channels, encoder_channels[0], 3, padding=1)

        self.encoder_stages = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        stage_channels: List[int] = [encoder_channels[0]]

        for idx, channels in enumerate(encoder_channels):
            self.encoder_stages.append(
                ResidualStage(channels, num_blocks=blocks_per_stage)
            )
            stage_channels.append(channels)
            if idx < len(encoder_channels) - 1:
                next_ch = encoder_channels[idx + 1]
                self.downs.append(
                    BiasFreeConv(channels, next_ch, kernel_size=3, stride=2, padding=1)
                )

        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = ResidualStage(
            bottleneck_channels, num_blocks=blocks_per_stage + 1, expansion=3
        )
        stage_channels.append(bottleneck_channels)

        self.up_projs = torch.nn.ModuleList()
        self.merge_convs = torch.nn.ModuleList()
        self.decoder_stages = torch.nn.ModuleList()

        decoder_channels = list(reversed(encoder_channels[:-1]))
        current_channels = bottleneck_channels
        for skip_ch in decoder_channels:
            self.up_projs.append(
                torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    BiasFreeConv(current_channels, skip_ch, kernel_size=3, padding=1),
                )
            )
            self.merge_convs.append(BiasFreeConv(skip_ch * 2, skip_ch, kernel_size=1))
            self.decoder_stages.append(
                ResidualStage(skip_ch, num_blocks=blocks_per_stage)
            )
            stage_channels.append(skip_ch)
            current_channels = skip_ch

        self.exit = BiasFreeConv(current_channels, image_channels, kernel_size=3, padding=1)

        self.conditioner = NoiseConditioner(
            stage_channels=stage_channels, descriptor_dim=self.prior_scales
        )

    def _apply_cond(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return x
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(cond, 2, dim=1)
        gamma = gamma[:, : x.shape[1]]
        beta = beta[:, : x.shape[1]]
        return x * (1.0 + gamma) + beta

    def forward(
        self, noisy: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_map, descriptor = self.prior_extractor(noisy)
        sigma = sigma.view(sigma.shape[0], 1)
        conds = iter(self.conditioner(sigma, descriptor))

        sigma_map = sigma.view(sigma.shape[0], 1, 1, 1).expand(
            -1, 1, noisy.shape[2], noisy.shape[3]
        )
        x = torch.cat([noisy, sigma_map, prior_map], dim=1)
        x = self.entry(x)
        x = self._apply_cond(x, next(conds, None))

        skips: List[torch.Tensor] = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x, next(conds, None))
            if idx < len(self.downs):
                skips.append(x)
                x = self.downs[idx](x)

        x = self.bottleneck(x, next(conds, None))

        for up, merge, stage in zip(
            self.up_projs, self.merge_convs, self.decoder_stages
        ):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = merge(x)
            x = stage(x, next(conds, None))

        residual = self.exit(x)
        denoised = torch.clamp(noisy - residual, 0.0, 1.0)
        return denoised, residual


def split_into_patches(
    tensor: torch.Tensor,
    patch_size: int,
    overlap: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("Overlap must be smaller than the patch size.")

    padded = torch.nn.functional.pad(tensor.unsqueeze(0), (8, 8, 8, 8), mode="reflect").squeeze(0)
    _, padded_h, padded_w = padded.shape

    positions_h = list(range(0, padded_h - patch_size, stride)) + [padded_h - patch_size]
    positions_w = list(range(0, padded_w - patch_size, stride)) + [padded_w - patch_size]

    patches = []
    coords = []
    for top in positions_h:
        for left in positions_w:
            patch = padded[:, top : top + patch_size, left : left + patch_size]
            patches.append(patch)
            coords.append((top, left))

    stacked = torch.stack(patches, 0)
    return stacked, coords, (padded_h, padded_w)


def assemble_from_patches(
    patches: torch.Tensor,
    coords: Sequence[Tuple[int, int]],
    padded_size: Tuple[int, int],
    patch_size: int,
    overlap: int,
) -> torch.Tensor:
    padded_h, padded_w = padded_size
    device = patches.device
    channels = patches.shape[1]
    result = torch.zeros(channels, padded_h, padded_w, device=device)
    weights = torch.zeros(1, padded_h, padded_w, device=device)

    if overlap > 0:
        window = torch.hann_window(patch_size, periodic=False, dtype=torch.float32, device=device)
        weight_patch = torch.outer(window, window).unsqueeze(0)
    else:
        weight_patch = torch.ones(1, patch_size, patch_size, device=device)

    for patch, (top, left) in zip(patches, coords):
        result[:, top : top + patch_size, left : left + patch_size] += patch * weight_patch
        weights[:, top : top + patch_size, left : left + patch_size] += weight_patch

    weights = weights.clamp(min=1e-6)
    result = result / weights
    result = result[:, 8:-8, 8:-8]
    return result


class ScriptPostprocessingCodeFormer(scripts_postprocessing.ScriptPostprocessing):
    name = "dePoison"
    order = 1100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None

    model_file = "Denoiser.safetensors"
    model_path = os.path.join(models_path, model_file)
    model_exists = os.path.isfile(model_path)

    def ui(self):
        with ui_components.InputAccordion(False, label="dePoison", visible=self.model_exists) as enable:
            with gr.Row():
                noise_level = gr.Slider(label="Noise level", minimum=0, maximum=255, step=1, value=25)
                iterations = gr.Slider(label="Iterations", minimum=1, maximum=4, step=1, value=1)
            with gr.Row():
                patch_size = gr.Slider(label="Patch size", minimum=128, maximum=1024, step=64, value=384)
                overlap = gr.Slider(label="Patch overlap", minimum=0, maximum=512, step=32, value=128)
            mode = gr.Radio(label="Mode", choices=["high quality", "fast"], value="fast")

        return {
            "enable": enable,
            "noise_level": noise_level,
            "iterations": iterations,
            "patch_size": patch_size,
            "overlap": overlap,
            "mode": mode,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, noise_level, iterations, patch_size, overlap, mode):
        if not enable or not self.model_exists:
            return

        if self.model is None:
            try:
                state_dict = load_safetensors(str(self.model_path))
                model_kwargs = self._infer_model_kwargs(state_dict)
                model = BiasFreeResidualDenoiser(**model_kwargs)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model = model
            except:
                print (f"[dePoison] failed to load model. Check {self.model_file} exists in {models_path}. ")
                return

        source_img = pp.image.convert("RGB")
        img = numpy.asarray(source_img, dtype=numpy.float32)[:, :, ::-1] / 255.0
        current = torch.from_numpy(img).permute(2, 0, 1)

        sigma_value = float(noise_level) / 255.0

        width, height = pp.image.size
        min_dim = min(width, height)
        patch_size = min(patch_size, min_dim+16)    # +16 for padding

        # tto_steps = 0
        # tto_tv_weight = 0.005
        # tto_lr = 0.05
        batch_size = 8
        iterations = int(iterations)
        patch_size = int(patch_size)
        overlap = int(overlap)

        stride = patch_size - overlap
        grid_h = len(list(range(0, height + 16 - patch_size, stride))) + 1
        grid_w = len(list(range(0, width + 16  - patch_size, stride))) + 1

        self.total_count = iterations * grid_h * grid_w * (len(QUALITY_GEOM_TRANSFORMS) * len(QUALITY_GAIN_FACTORS) if mode == "high quality" else 1)
        self.this_count = 0

        for _ in range(iterations):
            patches, coords, padded_size = split_into_patches(current, patch_size, overlap)
            patches = patches.to(self.device)
            effective_batch = max(1, min(batch_size, patches.size(0)))

            restored_batches = []
            with torch.no_grad():
                for start in range(0, patches.size(0), effective_batch):
                    batch = patches[start : start + effective_batch]
                    if mode == "high quality":
                        restored = self._quality_tta_pass(batch, sigma_value)
                    else:
                        restored = self._run_model(batch, sigma_value)
                    restored_batches.append(restored)

            restored_all = torch.cat(restored_batches, dim=0)
            assembled = assemble_from_patches(restored_all, coords, padded_size, patch_size, overlap)
            # if tto_steps > 0 and tto_tv_weight > 0.0 and tto_lr > 0.0:
                # current = self._apply_test_time_optimization(assembled, tto_steps, tto_tv_weight, tto_lr)
            # else:
            current = assembled.detach().cpu()

        print ("[dePoison] processed                 ")
        pp.image = tensor_to_image(current)


    def _run_model(self, batch: torch.Tensor, sigma_value: float) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        print (f"[dePoison] processing {self.this_count}/{self.total_count}", end="\r")
        self.this_count += batch.size(0)

        sigma_value = float(max(5e-4, min(1.0, sigma_value)))
        sigma = torch.full(
            (batch.shape[0], 1),
            sigma_value,
            dtype=batch.dtype,
            device=batch.device,
        )
        denoised, _ = self.model(batch, sigma)
        return denoised

    def _quality_tta_pass(self, batch: torch.Tensor, sigma_value: float) -> torch.Tensor:
        accum = torch.zeros_like(batch)
        total_weight = 0.0
        for rotation, flip in QUALITY_GEOM_TRANSFORMS:
            augmented = self._apply_geom_transform(batch, rotation, flip)
            for gain in QUALITY_GAIN_FACTORS:
                scaled = torch.clamp(augmented * gain, 0.0, 1.0)
                pred = self._run_model(scaled, sigma_value)
                if gain != 0.0:
                    pred = torch.clamp(pred / gain, 0.0, 1.0)
                pred = self._invert_geom_transform(pred, rotation, flip)
                accum = accum + pred
                total_weight += 1.0
        if total_weight == 0.0:
            return self._run_model(batch, sigma_value)
        return accum / total_weight


    @staticmethod
    def _apply_geom_transform(tensor: torch.Tensor, rotation: int, hflip: bool) -> torch.Tensor:
        result = tensor
        if rotation % 4:
            result = torch.rot90(result, k=rotation % 4, dims=(-2, -1))
        if hflip:
            result = torch.flip(result, dims=(-1,))
        return result

    @staticmethod
    def _invert_geom_transform(tensor: torch.Tensor, rotation: int, hflip: bool) -> torch.Tensor:
        result = tensor
        if hflip:
            result = torch.flip(result, dims=(-1,))
        inv_rot = (-rotation) % 4
        if inv_rot:
            result = torch.rot90(result, k=inv_rot, dims=(-2, -1))
        return result

    @staticmethod
    def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError("Expected tensor with shape (B,C,H,W) or (C,H,W) for TV computation.")
        dh = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        dw = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        return dh.abs().mean() + dw.abs().mean()

    def _apply_test_time_optimization(
        self,
        tensor: torch.Tensor,
        steps: int,
        tv_weight: float,
        lr: float,
    ) -> torch.Tensor:
        steps = max(0, int(steps))
        tv_weight = max(0.0, float(tv_weight))
        lr = max(0.0, float(lr))
        if steps == 0 or tv_weight == 0.0 or lr == 0.0:
            return tensor.detach().cpu()

        variable = tensor.detach().clone().requires_grad_(True)
        target = tensor.detach()
        optimizer = torch.optim.Adam([variable], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            tv = self._total_variation(variable)
            fidelity = torch.nn.functional.mse_loss(variable, target)
            loss = fidelity + tv_weight * tv
            loss.backward()
            optimizer.step()
            variable.data.clamp_(0.0, 1.0)

        return variable.detach().cpu()


    def _infer_model_kwargs(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        exit_weight = state_dict.get("exit.weight")
        if exit_weight is not None:
            kwargs["image_channels"] = exit_weight.shape[0]
        entry_weight = state_dict.get("entry.weight")
        if entry_weight is not None:
            kwargs["base_channels"] = entry_weight.shape[0]
        depth = self._count_indices(state_dict, "encoder_stages.")
        if depth:
            kwargs["depth"] = depth
        blocks = self._count_indices(state_dict, "encoder_stages.0.blocks.")
        if blocks:
            kwargs["blocks_per_stage"] = blocks
        if "residual_kernel_sizes" not in kwargs:
            kwargs["residual_kernel_sizes"] = self._infer_kernel_sizes(state_dict)
        kwargs.setdefault("image_channels", 3)
        kwargs.setdefault("base_channels", 48)
        kwargs.setdefault("depth", 3)
        kwargs.setdefault("blocks_per_stage", 2)
        kwargs.setdefault("residual_kernel_sizes", (3, 5, 7))
        return kwargs

    @staticmethod
    def _count_indices(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
        indices: set[int] = set()
        prefix_len = len(prefix)
        for key in state_dict.keys():
            if key.startswith(prefix):
                remainder = key[prefix_len:]
                parts = remainder.split(".")
                if parts and parts[0].isdigit():
                    indices.add(int(parts[0]))
        return len(indices)

    @staticmethod
    def _infer_kernel_sizes(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
        kernels: List[int] = []
        for key, tensor in state_dict.items():
            if key.startswith("prior_extractor.kernel_"):
                size = int(tensor.shape[-1])
                kernels.append(size)
        kernels.sort()
        return tuple(kernels) if kernels else (3, 5, 7)

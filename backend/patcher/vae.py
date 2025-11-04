import torch
import itertools

from tqdm import trange
from backend import memory_management
from backend.patcher.base import ModelPatcher


@torch.inference_mode()
def tiled_scale(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    dims = len(tile)
    B, _, H, W = samples.shape
    shapeF = [B, out_channels, round(H * upscale_amount), round(W * upscale_amount)]
    shape1 = [1, out_channels, round(H * upscale_amount), round(W * upscale_amount)]
    output = torch.empty(shapeF, device=output_device)

    for b in trange(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros(shape1, device=output_device)
        out_div = torch.zeros(shape1, device=output_device)

        for it in itertools.product(*(range(0, a[0], a[1] - overlap) for a in zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= ((1.0 / feather) * (t + 1))
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= ((1.0 / feather) * (t + 1))

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

        output[b:b + 1] = out / out_div
    return output


# this tiled decoder lightly modified from diffusers.models.autoencoders.autoencoder_kl.py
# faster than original (1 pass instead of 3) and better blending
@torch.inference_mode()
def tiled_decode_diffusers(samples, function, tile_x=64, tile_y=64, overlap=8, upscale=4, device="cpu"):
    def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    # Split samples into overlapping tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    rows = []
    for i in range(0, samples.shape[2], tile_y - overlap):
        row = []
        for j in range(0, samples.shape[3], tile_x - overlap):
            tile = samples[:, :, i : i + tile_y, j : j + tile_x]
            decoded = function(tile)
            row.append(decoded)
        rows.append(row)

    blend_extent = overlap * upscale
    row_limitX = (tile_x - overlap) * upscale
    row_limitY = (tile_y - overlap) * upscale
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile to the current tile and add the current tile to the result row
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limitY, :row_limitX])
        result_rows.append(torch.cat(result_row, dim=3))

    return torch.cat(result_rows, dim=2)


@torch.inference_mode()
def tiled_decode_DoE(samples, function, tile_x=64, tile_y=64, overlap=8, upscale=4, out_channels=3, device="cpu"):
    def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    o_tile_y = tile_y
    o_tile_x = tile_x
    output = torch.empty([samples.shape[0], out_channels, samples.shape[-2] * upscale, samples.shape[-1] * upscale], device=device)
    tile_x -= 3 * overlap
    tile_y -= 3 * overlap
    tile_x = min(samples.shape[-1], tile_x)
    tile_y = min(samples.shape[-2], tile_y)

    H = samples.shape[-2]
    W = samples.shape[-1]

    #decode only
    rows = []
    for b in trange(samples.shape[0]):
        tile_samples = torch.nn.functional.adaptive_avg_pool2d(samples[b:b+1], (o_tile_y, o_tile_x))
        y = 0
        while y < H:
            row = []
            x = 0
            while x < W:
                #latent positions, with padding top + bottom + left + right
                plys = y
                plye = min(y + tile_y + overlap, H)
                plxs = x
                plxe = min(x + tile_x + overlap, W)

                #overwrite padded tile, into correct position
                plys_p = min(int(o_tile_y * plys / H), o_tile_y - (plye-plys))
                plxs_p = min(int(o_tile_x * plxs / W), o_tile_x - (plxe-plxs))
                plye_p = plys_p + plye - plys
                plxe_p = plxs_p + plxe - plxs
                t = torch.empty_like(tile_samples)
                t.copy_(tile_samples)
                t[:, :, plys_p : plye_p, plxs_p : plxe_p] = samples[b:b+1, :, plys:plye, plxs:plxe]

                #VAE decode
                pixel_samples = function(t)

                #pixel positions, padding top + left only
                plye_p = plys_p + min(y + tile_y, H) - plys
                plxe_p = plxs_p + min(x + tile_x, W) - plxs
                lys_p = plys_p * upscale
                lxs_p = plxs_p * upscale
                lye_p = plye_p * upscale
                lxe_p = plxe_p * upscale

                p = pixel_samples[:, :, lys_p : lye_p, lxs_p : lxe_p]

                row.append(p)

                x += tile_x - overlap
            rows.append(row)
            y += tile_y - overlap

        blend_extent = overlap * upscale
        row_limitX = (tile_x - overlap) * upscale
        row_limitY = (tile_y - overlap) * upscale
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile to the current tile and add the current tile to the result row
                if i > 0:
                    tile = blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limitY, :row_limitX])
            result_rows.append(torch.cat(result_row, dim=3))

        output[b:b+1] = torch.cat(result_rows, dim=2)
    return output


class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False):
        if no_init:
            return

        self.memory_used_encode = lambda shape, dtype: (526 * shape[-2] * shape[-1]) * memory_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (64854 * shape[-2] * shape[-1]) * memory_management.dtype_size(dtype)
        if hasattr(model.config, "downscale_ratio"):
            self.downscale_ratio = int(model.config.downscale_ratio)
        elif hasattr(model.config, "scale_factor_spatial"):
            self.downscale_ratio = int(model.config.scale_factor_spatial)
        elif hasattr(model.config, "down_block_types"):
            self.downscale_ratio = int(2 ** (len(model.config.down_block_types) - 1))
        else:
            self.downscale_ratio = 8

        self.latent_channels = int(model.config.latent_channels)

        self.first_stage_model = model.eval()

        if device is None:
            device = memory_management.vae_device()

        self.device = device
        offload_device = memory_management.vae_offload_device()

        if dtype is None:
            dtype = memory_management.vae_dtype()

        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = memory_management.intermediate_device()

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        return n

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16, method="original"):
        if hasattr(self, "tile_info") and self.tile_info is not None:
            tile_x  = self.tile_info[0] // self.downscale_ratio
            tile_y  = self.tile_info[1] // self.downscale_ratio
            overlap = self.tile_info[2] // self.downscale_ratio
            method  = self.tile_info[3]

        decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).to(torch.float32)
        match method:
            case "diffusers":
                output = torch.clamp(tiled_decode_diffusers(samples, decode_fn, tile_x, tile_y, overlap, upscale=self.downscale_ratio, device=self.output_device) / 2.0, min=0.0, max=1.0)
            case "DoE":
                output = torch.clamp(tiled_decode_DoE(samples, decode_fn, tile_x, tile_y, overlap, upscale=self.downscale_ratio, device=self.output_device) / 2.0, min=0.0, max=1.0)
            case _:
                output = torch.clamp(
                    ((tiled_scale(samples, decode_fn, (tile_x // 2, tile_y * 2), overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                      tiled_scale(samples, decode_fn, (tile_x * 2, tile_y // 2), overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                      tiled_scale(samples, decode_fn, (tile_x, tile_y),          overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device))
                     / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        if hasattr(self, "tile_info") and self.tile_info is not None:
            tile_x  = self.tile_info[0]
            tile_y  = self.tile_info[1]
            overlap = self.tile_info[2]

        encode_fn = lambda a: self.first_stage_model.encode((2. * a - 1.).to(self.vae_dtype).to(self.device)).to(torch.float32)
        samples  = tiled_scale(pixel_samples, encode_fn, (tile_x, tile_y),          overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, (tile_x * 2, tile_y // 2), overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, (tile_x // 2, tile_y * 2), overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples /= 3.0
        return samples

    def decode(self, samples_in):
        do_tiled = False
        if memory_management.VAE_ALWAYS_TILED:
            do_tiled = True

        if not do_tiled:
            try:
                memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
                memory_management.load_models_gpu([self.patcher], memory_used)
                free_memory = memory_management.get_free_memory(self.device)
                batch_number = int(free_memory / memory_used)
                batch_number = max(1, batch_number)

                pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[-2] * self.downscale_ratio), round(samples_in.shape[-1] * self.downscale_ratio)), device=self.output_device)
                for x in range(0, samples_in.shape[0], batch_number):
                    samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(self.device)
                    pixel_samples[x:x + batch_number] = torch.clamp((self.first_stage_model.decode(samples).to(self.output_device).to(torch.float32) + 1.0) / 2.0, min=0.0, max=1.0)
            except memory_management.OOM_EXCEPTION:
                print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                do_tiled = True

        if do_tiled:
            if hasattr(self, "tile_info") and self.tile_info is not None:
                tile_x = self.tile_info[0] // self.downscale_ratio
                tile_y = self.tile_info[1] // self.downscale_ratio
            else:
                tile_x = 64
                tile_y = 64
            shape = (samples_in.shape[0], samples_in.shape[1], tile_y, tile_x)
            memory_used = self.memory_used_decode(shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_used)
            pixel_samples = self.decode_tiled(samples_in)

        return pixel_samples.to(self.output_device).movedim(1, -1)

    def encode(self, pixel_samples):
        pixel_samples = pixel_samples.movedim(-1, 1)

        do_tiled = False
        if memory_management.VAE_ALWAYS_TILED:
            do_tiled = True

        if not do_tiled:
            try:
                memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
                memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
                free_memory = memory_management.get_free_memory(self.device)
                batch_number = int(free_memory / memory_used)
                batch_number = max(1, batch_number)
                samples = torch.empty((pixel_samples.shape[0], self.latent_channels, round(pixel_samples.shape[-2] // self.downscale_ratio), round(pixel_samples.shape[-1] // self.downscale_ratio)), device=self.output_device)
                for x in range(0, pixel_samples.shape[0], batch_number):
                    pixels_in = (2. * pixel_samples[x:x + batch_number] - 1.).to(self.vae_dtype).to(self.device)
                    samples[x:x + batch_number] = self.first_stage_model.encode(pixels_in).to(self.output_device).to(torch.float32)

            except memory_management.OOM_EXCEPTION:
                print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
                do_tiled = True

        if do_tiled:
            if hasattr(self, "tile_info") and self.tile_info is not None:
                tile_x  = self.tile_info[0]
                tile_y  = self.tile_info[1]
            else:
                tile_x = 512
                tile_y = 512
            shape = (pixel_samples.shape[0], pixel_samples.shape[1], tile_y, tile_x)
            memory_used = self.memory_used_encode(shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
            samples = self.encode_tiled(pixel_samples)

        return samples

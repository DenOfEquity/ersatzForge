"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
import torch
import torch.nn as nn

from backend.state_dict import load_state_dict
from backend.utils import load_torch_file
from modules import devices, shared
from modules.paths_internal import models_path

sd_vae_taesd_models = {}


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1, n_gn = lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False), n_in*4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))
    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )


def Decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TAESDDecoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, decoder_path="taesd_decoder.pth", latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()

        midblock = False
        if latent_channels is None:
            if "taesd3" in str(decoder_path):
                latent_channels = 16
            elif "taef1" in str(decoder_path):
                latent_channels = 16
            elif "taef2" in str(decoder_path):
                latent_channels = 32
                midblock = True
            else:
                latent_channels = 4

        self.decoder = Decoder(latent_channels, use_midblock_gn=midblock)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))


class TAESDEncoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()

        midblock = False
        if latent_channels is None:
            if "taesd3" in str(encoder_path):
                latent_channels = 16
            elif "taef1" in str(encoder_path):
                latent_channels = 16
            elif "taef2" in str(encoder_path):
                latent_channels = 32
                midblock = True
            else:
                latent_channels = 4

        self.encoder = Encoder(latent_channels, use_midblock_gn=midblock)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class TAEHV(nn.Module):
    def __init__(self, latent_channels, decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True), encoder=False):
        super().__init__()
        self.image_channels = 3
        self.latent_channels = latent_channels

        if self.latent_channels == 16:  # Wan 2.1
            self.patch_size = 1
            act_func = nn.ReLU(inplace=True)
        elif self.latent_channels == 48:  # Wan 2.2
            self.patch_size = 2
            act_func = nn.ReLU(inplace=True)
        # else:  # HunyuanVideo 1.5
            # self.patch_size = 2
            # act_func = nn.LeakyReLU(0.2, inplace=True)

        if encoder:
            self.encoder = nn.Sequential(
                conv(self.image_channels*self.patch_size**2, 64), nn.ReLU(inplace=True),
                TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
                TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
                TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
                conv(64, self.latent_channels),
            )
            self.t_downscale = 2**sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        else:
            n_f = [256, 128, 64, 64]

            self.decoder = nn.Sequential(
                *(Clamp(), conv(self.latent_channels, n_f[0]), act_func),
                *(MemBlock(n_f[0], n_f[0], act_func), MemBlock(n_f[0], n_f[0], act_func), MemBlock(n_f[0], n_f[0], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False)),
                *(MemBlock(n_f[1], n_f[1], act_func), MemBlock(n_f[1], n_f[1], act_func), MemBlock(n_f[1], n_f[1], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False)),
                *(MemBlock(n_f[2], n_f[2], act_func), MemBlock(n_f[2], n_f[2], act_func), MemBlock(n_f[2], n_f[2], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False)),
                *(act_func, conv(n_f[3], self.image_channels * self.patch_size**2)),
            )

            self.t_upscale = 2 ** sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
            self.frames_to_trim = self.t_upscale - 1

    @staticmethod
    def apply_model_with_memblocks(model: nn.Sequential, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        for b in model:
            if isinstance(b, MemBlock):
                BT, C, H, W = x.shape
                T = BT // B
                _x = x.reshape(B, T, C, H, W)
                mem = nn.functional.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)

        BT, C, H, W = x.shape
        T = BT // B
        return x.view(B, T, C, H, W)


class TAEHVDecoder(nn.Module):
    def __init__(self, decoder_path: os.PathLike, latent_channels: int = None):
        super().__init__()

        if latent_channels is None:
            if "w2_1" in str(decoder_path):
                latent_channels = 16
            elif "w2_2" in str(decoder_path):
                latent_channels = 48
            else:
                latent_channels = 16

        self.decoder = TAEHV(latent_channels)
        load_state_dict(self.decoder, load_torch_file(decoder_path), ignore_start="encoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(2).movedim(2, 1)
        z = self.decoder.apply_model_with_memblocks(self.decoder.decoder, z)
        if self.decoder.patch_size > 1:
            z = nn.functional.pixel_shuffle(z, self.decoder.patch_size)
        return z[:, self.decoder.frames_to_trim :].squeeze(1)


class TAEHVEncoder(nn.Module):
    def __init__(self, encoder_path: os.PathLike, latent_channels: int = None):
        super().__init__()

        if latent_channels is None:
            if "w2_1" in str(encoder_path):
                latent_channels = 16
            elif "w2_2" in str(encoder_path):
                latent_channels = 48
            else:
                latent_channels = 16

        self.encoder = TAEHV(latent_channels, encoder=True)
        load_state_dict(self.encoder, load_torch_file(encoder_path), ignore_start="decoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)
        if self.encoder.patch_size > 1:
            z = nn.functional.pixel_unshuffle(z, self.encoder.patch_size)
        if z.shape[1] % self.encoder.t_downscale != 0:
            # pad at end to multiple of self.encoder.t_downscale
            n_pad = self.encoder.t_downscale - z.shape[1] % self.encoder.t_downscale
            padding = z[:, -1:].repeat_interleave(n_pad, dim=1)
            z = torch.cat([z, padding], 1)
        return self.encoder.apply_model_with_memblocks(self.encoder.encoder, z, parallel, show_progress_bar).squeeze(1)


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f'Downloading TAESD model to: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def decoder_model():
    v = False
    if shared.sd_model.is_sd3:
        model_name = "taesd3_decoder.pth"
    elif shared.sd_model.is_flux or shared.sd_model.is_lumina2:
        model_name = "taef1_decoder.pth"
    elif shared.sd_model.is_flux2:
        model_name = "taef2_decoder.pth"
    elif shared.sd_model.is_sdxl:
        if getattr(shared.sd_model, "is_mugen", False):
            model_name = "taef2_decoder.pth"
        else:
            model_name = "taesdxl_decoder.pth"
    elif shared.sd_model.is_sd1 or shared.sd_model.is_sd2:
        model_name = "taesd_decoder.pth"
    elif shared.sd_model.is_cosmos_predict2 or shared.sd_model.is_wan:
        v = True
        model_name = "taew2_1.pth" #to do: w2_2 (check number of latent channels in config)
        
    else:
        return None # preview can fall back to cheap approximation

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(models_path, "VAE-taesd", model_name)
        if v:
            download_model(model_path, 'https://github.com/madebyollin/taehv/raw/main/' + model_name)
        else:
            download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        if os.path.exists(model_path):
            loaded_model = (TAEHVDecoder if v else TAESDDecoder)(model_path)
            loaded_model.eval()
            loaded_model.to(devices.cpu, torch.float32)
            sd_vae_taesd_models[model_name] = loaded_model
            devices.torch_gc()
        else:
            raise FileNotFoundError('TAESD model not found')

    return loaded_model if v else loaded_model.decoder


def encoder_model():
    v = False
    if shared.sd_model.is_sd3:
        model_name = "taesd3_encoder.pth"
    elif shared.sd_model.is_flux or shared.sd_model.is_lumina2:
        model_name = "taef1_encoder.pth"
    elif shared.sd_model.is_flux2:
        model_name = "taef2_encoder.pth"
    elif shared.sd_model.is_sdxl:
        if getattr(shared.sd_model, "is_mugen", False):
            model_name = "taef2_encoder.pth"
        else:
            model_name = "taesdxl_encoder.pth"
    elif shared.sd_model.is_sd1 or shared.sd_model.is_sd2:
        model_name = "taesd_encoder.pth"
    elif shared.sd_model.is_cosmos_predict2 or shared.sd_model.is_wan:
        v = True
        model_name = "taew2_1.pth" #to do: w2_2
    else:
        raise FileNotFoundError('no TAESD encoder model for this architecture')

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(models_path, "VAE-taesd", model_name)
        download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        if os.path.exists(model_path):
            loaded_model = (TAEHVEncoder if v else TAESDEncoder)(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype_vae)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError('TAESD model not found')

    return loaded_model if v else loaded_model.encoder

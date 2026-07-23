"""Krea 2 (K2) — single-stream MMDiT.

Text tokens produced by a Qwen3-VL-4B 12-layer ``txtfusion`` adapter and patchified image tokens are
concatenated into one sequence and run through ``layers`` shared transformer blocks with
AdaLN-single modulation, GQA + per-head QK-norm + sigmoid-gated attention, SwiGLU MLP, and 3-axis RoPE.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.nn.flux import EmbedND, timestep_embedding, apply_rope
from backend.attention import attention_function
from backend.shared import global_variables


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention (scale stored zero-centered)."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = self.scale.to(device=x.device).to(torch.float32) + 1.0
        return F.rms_norm(x.to(torch.float32), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, negpip=None, freqs=None, mask=None, transformer_options={}):
        q = rearrange(self.wq(x), "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(self.wk(x), "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)

        v = rearrange(self.wv(x), "B L (H D) -> B H L D", H=self.kvheads)
        if negpip is not None:
            y_len = len(negpip)
            v[:, :, :y_len, :] *= negpip[:, None]

        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = attention_function(q, k, v, self.heads, mask=mask, skip_reshape=True)

        gate = self.gate(x)
        out.mul_(F.sigmoid(gate))
        return self.wo(out)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + self.lin.to(vec).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        out = vec + self.lin.to(vec)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None, transformer_options={}):
        x.add_(self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options))
        x.add_(self.mlp(self.postnorm(x)))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
            for _ in range(2)
        ])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
            for _ in range(2)
        ])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, negpip, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x.addcmul_(pregate, self.attn(torch.addcmul(preshift, 1 + prescale, self.prenorm(x)), negpip, freqs, mask, transformer_options=transformer_options))
        x.addcmul_(postgate, self.mlp(torch.addcmul(postshift, 1 + postscale, self.postnorm(x))))
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = torch.addcmul(shift, 1 + scale, self.norm(x))
        return self.linear(x)


class SingleStreamDiT(nn.Module):
    def __init__(self, features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
                 layers=28, patch=2, in_channels=16, bias=False, theta=1e3, txtlayers=12,
                 txtheads=20, txtkvheads=20, image_model=None, ctrl_patched=False, 
                 **kwargs):
        super().__init__()
        self.patch = patch
        self.ctrl_patched = ctrl_patched
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(16 * patch ** 2, features, bias=True)
        if self.ctrl_patched:
            self.first_ctrl = nn.Linear(32 * patch ** 2, features, bias=True)
        else:
            self.first_ctrl = None

        self.blocks = nn.ModuleList([
            SingleStreamBlock(features, heads, multiplier, bias, kvheads)
            for _ in range(layers)
        ])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, 16)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

    def forward(self, x, timesteps, context, negpip=None, attention_mask=None, transformer_options={}, **kwargs):
        bs, _, H_orig, W_orig = x.shape
        device = x.device

        patch = self.patch
        if patch >= 2:
            pad_h = (patch - H_orig % patch) % patch
            pad_w = (patch - W_orig % patch) % patch
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        start_sigma = getattr(global_variables, "krea2_control_lora_start_sigma", 0.0) or 0.0
        end_sigma = getattr(global_variables, "krea2_control_lora_stop_sigma", 1.0)
        if end_sigma is None:
            end_sigma = 1.0
        strength = getattr(global_variables, "krea2_control_lora_strength", 0.0) or 0.0
        fidelity = getattr(global_variables, "krea2_control_lora_fidelity", 1.0) or 1.0
        if timesteps[0].item() <= start_sigma and timesteps[0].item() > end_sigma and strength > 0.0:
            ctrl = getattr(global_variables, "krea2_control_lora_latent", None)
            setattr(global_variables, "krea2_control_lora_apply", strength)
        else:
            ctrl = None
            setattr(global_variables, "krea2_control_lora_apply", 0.0)

        ctrlimg = None
        ctrlpos = None
        if ctrl is None:
            img = self.first(x)
        else:
            if patch >= 2:
                pad_h = (patch - ctrl.shape[-2] % patch) % patch
                pad_w = (patch - ctrl.shape[-1] % patch) % patch
                ctrl = torch.nn.functional.pad(ctrl, (0, pad_w, 0, pad_h), mode="circular")
                ch = ctrl.shape[-1] // patch
                cw = ctrl.shape[-2] // patch

            ctrl = rearrange(ctrl, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
            if bs > 1:
                ctrl = ctrl.repeat(bs, 1, 1)

            if self.ctrl_patched: # Control LoRA: concat along channels dim (ref must be same size as noise)
                img = torch.cat([x, ctrl], dim=2)
                img = self.first_ctrl(img)
            else:                 # reference: all ctrl stuff could be cached; concat on dim 1, so ref could be different size to noise
                img = self.first(x)
                ctrlimg = self.first(ctrl)
                ctrlids = torch.zeros(ch, cw, 3, device=device, dtype=torch.float32)
                ctrlids[..., 0] = 1
                ctrlids[..., 1] = torch.arange(ch, device=device, dtype=torch.float32)[:, None]
                ctrlids[..., 2] = torch.arange(cw, device=device, dtype=torch.float32)[None, :]
                ctrlpos = ctrlids.reshape(1, ch * cw, 3).repeat(bs, 1, 1)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        # Position ids: text at 0, image at (0, h_idx, w_idx).
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
        imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
        imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)

        if ctrlimg is None:
            ctrllen = 0
            combined = torch.cat((context, img), dim=1)
            pos = torch.cat((txtpos, imgpos), dim=1)
            attn_bias = None
        else:
            ctrllen = ctrlimg.shape[1]
            combined = torch.cat((context, ctrlimg, img), dim=1)
            pos = torch.cat((txtpos, ctrlpos, imgpos), dim=1)
            if fidelity == 1.0:
                attn_bias = None
            else:
                rows = txtlen + ctrllen
                cols = torch.arange(txtlen, txtlen + ctrllen, device=combined.device)
                L = rows + imglen
                attn_bias = torch.zeros(1, 1, L, L, device=combined.device, dtype=combined.dtype)
                attn_bias[:, :, rows:, cols] = math.log(max(fidelity, 1e-4))

        freqs = self.pe_embedder(pos)

        if negpip is not None:
            negpip = negpip[0]

        for block in self.blocks:
            combined = block(combined, negpip, tvec, freqs, attn_bias, transformer_options=transformer_options)

        final = self.last(combined, t)
        out = final[:, txtlen+ctrllen:]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch)
        out = out[:, :, :H_orig, :W_orig]  # crop padding back off
        return out

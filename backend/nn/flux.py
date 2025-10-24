# Single File Implementation of Flux with aggressive optimizations, Copyright Forge 2024
# If used outside Forge, only non-commercial use is allowed.
# See also https://github.com/black-forest-labs/flux


import math
import torch

from torch import nn
from einops import rearrange, repeat
from backend.attention import attention_function
from backend.utils import fp16_fix


def attention(q, k, v, pe):
    _shape = q.shape # k.shape is always the same
    _reshape = list(_shape[:-1]) + [-1, 1, 2]

    q = q.to(torch.float32).reshape(*_reshape)
    q = torch.add(torch.mul(pe[..., 0], q[..., 0]), torch.mul(pe[..., 1], q[..., 1]))
    q = q.reshape(*_shape).type_as(v)

    k = k.to(torch.float32).reshape(*_reshape)
    k = torch.add(torch.mul(pe[..., 0], k[..., 0]), torch.mul(pe[..., 1], k[..., 1]))
    k = k.reshape(*_shape).type_as(v)
    
    x = attention_function(q, k, v, q.shape[1], skip_reshape=True)
    return x


def rope(pos, dim, theta):
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)

    # out = torch.einsum("...n,d->...nd", pos, omega)
    out = pos.unsqueeze(-1) * omega.unsqueeze(0)

    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    del omega, cos_out, sin_out

    # out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    b, n, d, _ = out.shape
    out = out.view(b, n, d, 2, 2)

    return out.to(torch.float32)


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.to(torch.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xq_out = xq_out.reshape(*xq.shape).type_as(xq)
    del xq, xq_
    xk_ = xk.to(torch.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    xk_out = xk_out.reshape(*xk.shape).type_as(xk)
    del xk, xk_
    return xq_out, xk_out


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2

    # TODO: Once A trainer for flux get popular, make timestep_embedding consistent to that trainer

    # Do not block CUDA stream, but having about 1e-4 differences with Flux official codes:
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    # Block CUDA stream, but consistent with official codes:
    # freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].to(torch.float32) * freqs[None]
    del freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    del args
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class EmbedND(nn.Module):
    def __init__(self, theta, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        del ids, n_axes
        return emb.unsqueeze(1)

# dynamic PE
#options that seems relevant is dynamic YaRN - non-dynamic is bad, NTK less good
import numpy as np
def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base)) #Inverse dim formula to find number of rotations


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """
    Find the correction range for NTK-by-parts interpolation.
    """
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim-1) #Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001 #Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base, scale):
    """
    Calculate the new base for NTK-aware scaling.
    """
    return base * (scale ** (dim / (dim - 2)))

from typing import Union, List
def get_1d_rotary_pos_embed(
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        linear_factor=1.0,
        ntk_factor=1.0,
        freqs_dtype=torch.float32,
        yarn=False,
        max_pe_len=None,
        ori_max_pe_len=64,
        dype=False,
        current_timestep=1.0,
):
    """
    Precompute the frequency tensor for complex exponentials with RoPE.
    Supports YARN interpolation for vision transformers.

    Args:
        dim (`int`):
            Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`):
            Position indices for the frequency tensor. [S] or scalar.
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for linear interpolation.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for NTK-Aware RoPE.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            Data type of the frequency tensor.
        yarn (`bool`, *optional*, defaults to False):
            If True, use YARN interpolation combining NTK, linear, and base methods.
        max_pe_len (`int`, *optional*):
            Maximum position encoding length (current patches for vision models).
        ori_max_pe_len (`int`, *optional*, defaults to 64):
            Original maximum position encoding length (base patches for vision models).
        dype (`bool`, *optional*, defaults to False):
            If True, enable DyPE (Dynamic Position Encoding) with timestep-aware scaling.
        current_timestep (`float`, *optional*, defaults to 1.0):
            Current timestep for DyPE, normalized to [0, 1] where 1 is pure noise.

    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
            returns tuple of (cos, sin) tensors.
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))

        freqs_linear = 1.0 / torch.einsum(
            '..., f -> ... f',
            scale,
            (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        )

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(
            new_base,
            (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)
        )
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        if dype:
            beta_0 = beta_0 ** (2.0 * (current_timestep ** 2.0))
            beta_1 = beta_1 ** (2.0 * (current_timestep ** 2.0))

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        if dype:
            gamma_0 = gamma_0 ** (2.0 * (current_timestep ** 2.0))
            gamma_1 = gamma_1 ** (2.0 * (current_timestep ** 2.0))

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

    else:
        theta_ntk = theta * ntk_factor
        freqs = 1.0 / (theta_ntk ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)) / linear_factor

    freqs = torch.outer(pos.squeeze(0), freqs)

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.to(torch.float32)

    freqs_cos = freqs.cos().to(torch.float32)
    freqs_sin = freqs.sin().to(torch.float32)

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        mscale = torch.where(scale <= 1., torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0).to(scale)
        freqs_cos = freqs_cos * mscale
        freqs_sin = freqs_sin * mscale

    return freqs_cos, freqs_sin


class FluxPosEmbed(nn.Module):
    def __init__(
            self,
            theta: int,
            axes_dim: List[int],
            base_resolution: int = 1024,
            method: str = 'yarn',
            dype: bool = True,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.base_resolution = base_resolution
        self.patch_size = 16
        self.base_patches = self.base_resolution // self.patch_size
        self.method = method
        self.dype = dype if method != 'base' else False
        self.current_timestep = 1.0

    def set_timestep(self, timestep: float):
        """Set current timestep for DyPE. Timestep normalized to [0, 1] where 1 is pure noise."""
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        for i in range(n_axes):
            common_kwargs = {
                'dim': self.axes_dim[i],
                'pos': pos[:, i],
                'theta': self.theta,
                'freqs_dtype': freqs_dtype,
            }

            if i > 0:
                max_pos = pos[:, i].max().item()
                current_patches = max_pos + 1

                if self.method == 'yarn' and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        yarn=True,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=self.base_patches,
                        dype=self.dype,
                        current_timestep=self.current_timestep,
                    )

                elif self.method == 'ntk' and current_patches > self.base_patches:
                    base_ntk = (current_patches / self.base_patches) ** (
                            self.axes_dim[i] / (self.axes_dim[i] - 2)
                    )
                    ntk_factor = base_ntk ** (2.0 * (self.current_timestep ** 2.0)) if self.dype else base_ntk
                    ntk_factor = max(1.0, ntk_factor)

                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, ntk_factor=ntk_factor)

                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin
## end dynamic PE


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x = self.silu(self.in_layer(x))
        return self.out_layer(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        self.scale.data = self.scale.data.to(x.device, x.dtype)

        if x.dtype in [torch.bfloat16, torch.float32]:
            n = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6) * self.scale
        else:
            n = torch.rsqrt(torch.mean(x.to(torch.float32) ** 2, dim=-1, keepdim=True) + 1e-6).to(x.dtype) * self.scale
        return x.mul(n)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(k), k


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pe):
        qkv = self.qkv(x)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k)

        x = attention(q, k, v, pe=pe)
        del q, k, v

        x = self.proj(x)
        return x


class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, scratchQ, scratchK, scratchV, img, txt, vec, pe):
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = torch.addcmul(img_mod1_shift, img_mod1_scale.add_(1), img_modulated)
        del img_mod1_shift, img_mod1_scale
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated

        T = txt.shape[1]

        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = img_qkv.shape
        H = self.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        scratchQ[:, : , T:, :], scratchK[:, : , T:, :], scratchV[:, : , T:, :] = img_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        del img_qkv

        scratchQ[:, : , T:, :], scratchK[:, : , T:, :] = self.img_attn.norm(scratchQ[:, : , T:, :], scratchK[:, : , T:, :])

        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)
        del vec

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = torch.addcmul(txt_mod1_shift, txt_mod1_scale.add_(1), txt_modulated)
        del txt_mod1_shift, txt_mod1_scale
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated

        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = txt_qkv.shape
        scratchQ[:, : , :T, :], scratchK[:, : , :T, :], scratchV[:, : , :T, :] = txt_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        del txt_qkv

        scratchQ[:, : , :T, :], scratchK[:, : , :T, :] = self.txt_attn.norm(scratchQ[:, : , :T, :], scratchK[:, : , :T, :])

        attn = attention(scratchQ, scratchK, scratchV, pe=pe)
        del pe
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        del attn

        img.addcmul_(img_mod1_gate, self.img_attn.proj(img_attn))
        del img_attn, img_mod1_gate
        img.addcmul_(img_mod2_gate, self.img_mlp(torch.addcmul(img_mod2_shift, img_mod2_scale.add_(1), self.img_norm2(img))))
        del img_mod2_gate, img_mod2_scale, img_mod2_shift

        txt.addcmul_(txt_mod1_gate, self.txt_attn.proj(txt_attn))
        del txt_attn, txt_mod1_gate
        txt.addcmul_(txt_mod2_gate, self.txt_mlp(torch.addcmul(txt_mod2_shift, txt_mod2_scale.add_(1), self.txt_norm2(txt))))
        del txt_mod2_gate, txt_mod2_scale, txt_mod2_shift

        txt = fp16_fix(txt)

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, scratchA, x, vec, pe):
        mod_shift, mod_scale, mod_gate = self.modulation(vec)
        del vec
        x_mod = torch.addcmul(mod_shift, mod_scale.add_(1), self.pre_norm(x))
        del mod_shift, mod_scale
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        del x_mod

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k)
        scratchA[:, :, :3072] = attention(q, k, v, pe=pe)
        scratchA[:, :, 3072:] = self.mlp_act(mlp)
        del q, k, v, pe
        output = self.linear2(scratchA)

        x.addcmul_(mod_gate, output)
        del mod_gate, output

        x = fp16_fix(x)

        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        del vec
        x = torch.addcmul(shift[:, None, :], scale.add_(1)[:, None, :], self.norm_final(x))
        del scale, shift
        x = self.linear(x)
        return x


class IntegratedFluxTransformer2DModel(nn.Module):
    def __init__(self, in_channels: int, vec_in_dim: int, context_in_dim: int, hidden_size: int, mlp_ratio: float, num_heads: int, depth: int, depth_single_blocks: int, axes_dim: list[int], theta: int, qkv_bias: bool, guidance_embed: bool):
        super().__init__()

        self.guidance_embed = guidance_embed
        self.in_channels = in_channels * 4
        self.out_channels = self.in_channels

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(theta=theta, axes_dim=axes_dim)
        # self.pe_embedder = FluxPosEmbed(theta=theta, axes_dim=axes_dim, )

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        device = img.device
        dtype = img.dtype

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec.add_(self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype)))
        vec.add_(self.vector_in(y))
        txt = self.txt_in(txt)
        del y, guidance

        ids = torch.cat((txt_ids, img_ids), dim=1)
        del txt_ids, img_ids
        pe = self.pe_embedder(ids)
        
        # pes = []
        # for i in range(ids.shape[0]):
            # pe = self.pe_embedder(ids[i])

            # out = torch.stack([pe[0], -pe[1], pe[1], pe[0]], dim=-1).unsqueeze(0)
            # out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
            # pes.append(out.unsqueeze(1))
        # pe = torch.cat(pes, dim=0)

        del ids

        scratchQ = torch.empty((img.shape[0], 24, img.shape[1]+txt.shape[1], 128), device=device, dtype=dtype)   # preallocated for combined q|k|v_img|txt
        scratchK = torch.empty((img.shape[0], 24, img.shape[1]+txt.shape[1], 128), device=device, dtype=dtype)
        scratchV = torch.empty((img.shape[0], 24, img.shape[1]+txt.shape[1], 128), device=device, dtype=dtype)
        for block in self.double_blocks:
            img, txt = block(scratchQ, scratchK, scratchV, img=img, txt=txt, vec=vec, pe=pe)
        img = torch.cat((txt, img), 1)

        scratchA = torch.empty((img.shape[0], img.shape[1], 15360), device=device, dtype=dtype)
        for block in self.single_blocks:
            img = block(scratchA, img, vec=vec, pe=pe)
        del pe
        img = img[:, txt.shape[1]:, ...]
        del txt
        img = self.final_layer(img, vec)
        del vec
        return img

    def forward(self, x, timestep, context, y, guidance=None, **kwargs):
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype
        patch_size = 2
        pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
        pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        del x, pad_h, pad_w
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
        del input_device, input_dtype
        out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance)
        del img, img_ids, txt_ids, timestep, context
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        del h_len, w_len, bs
        return out

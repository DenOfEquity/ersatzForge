# Single File Implementation of Flux with aggressive optimizations, Copyright Forge 2024
# If used outside Forge, only non-commercial use is allowed.
# See also https://github.com/black-forest-labs/flux

# modified for Flux2 Klein 4B/9B
# additional ref: ForgeNeo


from dataclasses import dataclass
import math
import torch
from torch import nn
from einops import rearrange, repeat

from backend.attention import attention_function
from backend.utils import fp16_fix

from modules import shared

from .flux import attention, rope, apply_rope, timestep_embedding
from .flux import EmbedND, RMSNorm, QKNorm


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, bias=True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x):
        x = self.silu(self.in_layer(x))
        return self.out_layer(x)


class YakMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def build_mlp(hidden_size: int, mlp_hidden_dim: int, mlp_silu_act: bool = False, yak_mlp: bool = False) -> nn.Module:
    if yak_mlp:
        return YakMLP(hidden_size, mlp_hidden_dim)
    if mlp_silu_act:
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )
    else:
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

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


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim, double, bias=True):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=bias)

    def forward(self, vec):
        if vec.ndim == 2:
            vec = vec.unsqueeze(1)
        out = self.lin(nn.functional.silu(vec)).chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, modulation=True, mlp_silu_act=False, proj_bias=True, yak_mlp=False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.modulation = modulation

        if self.modulation:
            self.img_mod = Modulation(hidden_size, double=True)

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.img_mlp = build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=mlp_silu_act, yak_mlp=yak_mlp)

        if self.modulation:
            self.txt_mod = Modulation(hidden_size, double=True)

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.txt_mlp = build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=mlp_silu_act, yak_mlp=yak_mlp)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = torch.addcmul(img_mod1.shift, 1 + img_mod1.scale, img_modulated)
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, txt_modulated)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del txt_qkv
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v
        # run actual attention
        attn = attention(q, k, v, pe=pe)
        del q, k, v

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img.addcmul_(self.img_attn.proj(img_attn), img_mod1.gate)
        del img_attn
        img.addcmul_(img_mod2.gate, self.img_mlp(torch.addcmul(img_mod2.shift, 1 + img_mod2.scale, self.img_norm2(img))))

        # calculate the txt blocks
        txt.addcmul_(self.txt_attn.proj(txt_attn), txt_mod1.gate)
        del txt_attn
        txt.addcmul_(txt_mod2.gate, self.txt_mlp(torch.addcmul(txt_mod2.shift, 1 + txt_mod2.scale, self.txt_norm2(txt))))

        txt = fp16_fix(txt)

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float=4.0, qk_scale=None, modulation=True, mlp_silu_act=False, bias=True, yak_mlp=False):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp_hidden_dim_first = self.mlp_hidden_dim
        self.yak_mlp = yak_mlp
        if mlp_silu_act:
            self.mlp_hidden_dim_first = int(hidden_size * mlp_ratio * 2)
            self.mlp_act = SiLUActivation()
        else:
            self.mlp_act = nn.GELU(approximate="tanh")

        if self.yak_mlp:
            self.mlp_hidden_dim_first *= 2
            self.mlp_act = nn.SiLU()

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim_first, bias=bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=bias)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if modulation:
            self.modulation = Modulation(hidden_size, double=False)
        else:
            self.modulation = None

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        if self.modulation:
            mod, _ = self.modulation(vec)
        else:
            mod = vec

        x_mod = torch.addcmul(mod.shift, 1 + mod.scale, self.pre_norm(x))

        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        q, k = self.norm(q, k)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        del q, k, v
        # compute activation in mlp stream, cat again and run second linear layer
        if self.yak_mlp:
            mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2 :]) * mlp[..., : self.mlp_hidden_dim_first // 2]
        else:
            mlp = self.mlp_act(mlp)
        output = self.linear2(torch.cat((attn, mlp), 2))
        x.addcmul_(mod.gate, output)

        x = fp16_fix(x)

        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, bias=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=bias))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = torch.addcmul(shift, scale.add_(1), self.norm_final(x))
        x = self.linear(x)
        return x


class IntegratedFlux2Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vec_in_dim: int,
        context_in_dim: int,
        hidden_size: int,
        mlp_ratio: float,
        num_heads: int,
        depth: int,
        depth_single_blocks: int,
        axes_dim: list[int],
        theta: int,
        patch_size: int,
        qkv_bias: bool,
        guidance_embed: bool,
        txt_ids_dims: list[int],
        global_modulation: bool = False,
        mlp_silu_act: bool = False,
        ops_bias: bool = True,
        default_ref_method: str = "offset",
        ref_index_scale: float = 1.0,
        yak_mlp: bool = False,
        txt_norm: bool = False,
    ):
        super().__init__()

        self.guidance_embed = guidance_embed
        self.patch_size = patch_size
        self.in_channels = in_channels * patch_size * patch_size
        self.out_channels = out_channels * patch_size * patch_size

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.axes_dim = axes_dim

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=ops_bias)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=ops_bias)
        if vec_in_dim is not None:
            self.vec_in_dim = vec_in_dim
            self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=ops_bias) if guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size, bias=ops_bias)

        self.txt_norm = RMSNorm(context_in_dim) if txt_norm else None

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    modulation=global_modulation is False,
                    mlp_silu_act=mlp_silu_act,
                    proj_bias=ops_bias,
                    yak_mlp=yak_mlp,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    modulation=global_modulation is False,
                    mlp_silu_act=mlp_silu_act,
                    bias=ops_bias,
                    yak_mlp=yak_mlp,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, bias=ops_bias)

        if global_modulation:
            self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, bias=False)
            self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, bias=False)
            self.single_stream_modulation = Modulation(self.hidden_size, double=False, bias=False)

        self.txt_ids_dims = txt_ids_dims
        self.global_modulation = global_modulation
        # self.default_ref_method = default_ref_method
        # self.ref_index_scale = ref_index_scale

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
    ) -> torch.Tensor:

        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if hasattr(self, "vec_in_dim"):
            if y is None:
                y = torch.zeros((img.shape[0], self.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, : self.vec_in_dim])

        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        vec_orig = vec
        if self.global_modulation:
            vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(vec_orig))

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = fp16_fix(img)
        img = torch.cat((txt, img), 1)

        if self.global_modulation:
            vec, _ = self.single_stream_modulation(vec_orig)

        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)

        img = img[:, txt.shape[1] :, ...]

        return self.final_layer(img, vec_orig)  # (N, T, patch_size ** 2 * out_channels)

    def process_img(self, x, index=0):
        bs, c, h, w = x.shape
        patch_size = self.patch_size

        if patch_size >= 2:
            pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
            pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        steps_h = h_len
        steps_w = w_len

        img_ids = torch.zeros((steps_h, steps_w, len(self.axes_dim)), device=x.device, dtype=torch.float32)
        img_ids[:, :, 0] = index
        img_ids[:, :, 1] += torch.linspace(0, h_len - 1, steps=steps_h, device=x.device, dtype=torch.float32).unsqueeze(1)
        img_ids[:, :, 2] += torch.linspace(0, w_len - 1, steps=steps_w, device=x.device, dtype=torch.float32).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size
        img, img_ids = self.process_img(x)
        img_tokens = img.shape[1]

        index = 0
        if hasattr(shared, "klein_latents"):
            for ref, str in zip(getattr(shared, "klein_latents"), getattr(shared, "klein_strength")):
                if ref is not None and str > 0.0:
                    index += 10
                    klein, klein_ids = self.process_img(ref, index=index)
                    klein *= str
                    img = torch.cat([img, klein.to(img)], dim=1)
                    img_ids = torch.cat([img_ids, klein_ids.to(img_ids)], dim=1)

        txt_ids = torch.zeros((bs, context.shape[1], len(self.axes_dim)), device=x.device, dtype=torch.float32)

        if len(self.txt_ids_dims) > 0:
            for i in self.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control)
        out = out[:, :img_tokens]

        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)[:, :, :h_orig, :w_orig]

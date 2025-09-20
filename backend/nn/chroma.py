# implementation of Chroma for Forge, inspired by https://github.com/lodestone-rock/ComfyUI_FluxMod

import math
import torch

from torch import nn
from einops import rearrange, repeat
from backend.utils import fp16_fix
from backend.attention import attention_function

from .flux import timestep_embedding, EmbedND, MLPEmbedder, RMSNorm, QKNorm, SelfAttention


class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range( n_layers)])
        self.norms = nn.ModuleList([RMSNorm( hidden_dim) for x in range( n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, img_mod1, img_mod2, txt_mod1, txt_mod2, pe):
        i1_shift, i1_scale, i1_gate = torch.split(img_mod1, 1, dim=1)
        i2_shift, i2_scale, i2_gate = torch.split(img_mod2, 1, dim=1)
        t1_shift, t1_scale, t1_gate = torch.split(txt_mod1, 1, dim=1)
        t2_shift, t2_scale, t2_gate = torch.split(txt_mod2, 1, dim=1)

        img_modulated = torch.addcmul(i1_shift, i1_scale.add_(1), self.img_norm1(img))
        img_qkv = self.img_attn.qkv(img_modulated)
        B, L, _ = img_qkv.shape
        H = self.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = torch.addcmul(t1_shift, t1_scale.add_(1), txt_modulated)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        B, L, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        del txt_qkv

        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        _shape = q.shape # k.shape is always the same
        _reshape = list(_shape[:-1]) + [-1, 1, 2]
        q = q.to(torch.float32).reshape(*_reshape)
        q = torch.add(torch.mul(pe[..., 0], q[..., 0]), torch.mul(pe[..., 1], q[..., 1]))
        q = q.reshape(*_shape).type_as(v)

        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        k = k.to(torch.float32).reshape(*_reshape)
        k = torch.add(torch.mul(pe[..., 0], k[..., 0]), torch.mul(pe[..., 1], k[..., 1]))
        k = k.reshape(*_shape).type_as(v)
        del pe

        attn = attention_function(q, k, v, q.shape[1], skip_reshape=True)
        del q, k, v

        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        img.addcmul_(i1_gate, self.img_attn.proj(img_attn))
        img.addcmul_(i2_gate, self.img_mlp(torch.addcmul(i2_shift, i2_scale.add_(1), self.img_norm2(img))))
        txt.addcmul_(t1_gate, self.txt_attn.proj(txt_attn))
        txt.addcmul_(t2_gate, self.txt_mlp(torch.addcmul(t2_shift, t2_scale.add_(1), self.txt_norm2(txt))))

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

    def forward(self, x, shift, scale, gate, pe):
        x_mod = torch.addcmul(shift, scale.add_(1), self.pre_norm(x))
        del shift, scale
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        del x_mod

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        del qkv

        q, k = self.norm(q, k)

        _shape = q.shape # k.shape is always the same
        _reshape = list(_shape[:-1]) + [-1, 1, 2]

        q = q.to(torch.float32).reshape(*_reshape)
        q = torch.add(torch.mul(pe[..., 0], q[..., 0]), torch.mul(pe[..., 1], q[..., 1]))
        q = q.reshape(*_shape).type_as(v)

        k = k.to(torch.float32).reshape(*_reshape)
        k = torch.add(torch.mul(pe[..., 0], k[..., 0]), torch.mul(pe[..., 1], k[..., 1]))
        k = k.reshape(*_shape).type_as(v)

        del pe
        
        attn = attention_function(q, k, v, q.shape[1], skip_reshape=True)
        del q, k, v

        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        del attn, mlp

        x.addcmul_(gate, output)
        del gate, output

        x = fp16_fix(x)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, shift, scale):
        x = torch.addcmul(shift, scale.add_(1), self.norm_final(x))
        x = self.linear(x)
        return x


class IntegratedChromaTransformer2DModel(nn.Module):
    def __init__(self, in_channels: int, vec_in_dim: int, context_in_dim: int, hidden_size: int, mlp_ratio: float, num_heads: int, depth: int, depth_single_blocks: int, axes_dim: list[int], theta: int, qkv_bias: bool, guidance_embed):
        super().__init__()
        
        self.in_channels = in_channels * 4
        self.out_channels = self.in_channels

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.distilled_guidance_layer = Approximator(64, 3072, 5120, 5)
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
        
    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps, guidance=None):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)
        device = img.device
        dtype = img.dtype
        nb_double_block = len(self.double_blocks)
        nb_single_block = len(self.single_blocks)
        
        mod_index_length = nb_double_block*12 + nb_single_block*3 + 2
        distill_timestep = timestep_embedding(timesteps, 16).to(device=device, dtype=dtype) # this was timesteps.detach().clone()
        distill_guidance = timestep_embedding(guidance, 16).to(device=device, dtype=dtype) # this was guidance.detach().clone()
        del guidance
        modulation_index = timestep_embedding(torch.arange(mod_index_length), 32).to(device=device, dtype=dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        timestep_guidance = torch.cat([distill_timestep, distill_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
        del distill_timestep, distill_guidance
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
        del timestep_guidance, modulation_index
        mod_vectors = self.distilled_guidance_layer(input_vec)
        del input_vec
        
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        del txt_ids, img_ids
        pe = self.pe_embedder(ids)
        del ids

        idx_i = 3 * nb_single_block
        idx_t = 3 * nb_single_block + 6 * nb_double_block
        for i, block in enumerate(self.double_blocks):
            img_mod1 = mod_vectors[:, idx_i+0:idx_i+3, :]
            img_mod2 = mod_vectors[:, idx_i+3:idx_i+6, :]
            idx_i += 6
            txt_mod1 = mod_vectors[:, idx_t+0:idx_t+3, :]
            txt_mod2 = mod_vectors[:, idx_t+3:idx_t+6, :]
            idx_t += 6
            img, txt = block(img=img, txt=txt, img_mod1=img_mod1, img_mod2=img_mod2, txt_mod1=txt_mod1, txt_mod2=txt_mod2, pe=pe)
        img = torch.cat((txt, img), 1)

        idx = 0
        for i, block in enumerate(self.single_blocks):
            img = block(img, shift=mod_vectors[:, idx+0:idx+1, :], scale=mod_vectors[:, idx+1:idx+2, :], gate=mod_vectors[:, idx+2:idx+3, :], pe=pe)
            idx += 3
        del pe
        img = img[:, txt.shape[1]:, ...]

        idx = 3 * nb_single_block + 12 * nb_double_block
        img = self.final_layer(img, mod_vectors[:, idx:idx+1, :], mod_vectors[:, idx+1:idx+2, :])

        return img

    def forward(self, x, timestep, context, guidance=None, **kwargs):
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
        out = self.inner_forward(img, img_ids, context, txt_ids, timestep, guidance)
        del img, img_ids, txt_ids, timestep, context
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        del h_len, w_len, bs
        return out

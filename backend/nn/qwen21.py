# https://github.com/huggingface/diffusers (Apache 2.0) Qwen-Image 2.1
# from there to ComfyUI
# from there to here, with some modifications/cleanups


import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.attention import attention_function

from .flux import EmbedND, timestep_embedding

from modules import shared


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, sample_proj_bias=True):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = F.silu(sample)
        sample = self.linear_2(sample)
        return sample


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False)

    def forward(self, timestep, dtype):
        return self.timestep_embedder(timestep_embedding(timestep.to(torch.float32), 256).to(dtype))


class ZeroCenteredRMSNorm(nn.Module):
    # stored weight is scale - 1, applied in fp32
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x):
        w = self.weight.to(device=x.device).to(torch.float32) + 1.0
        return F.rms_norm(x.to(torch.float32), w.shape, weight=w, eps=self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, fused=True):
        super().__init__()
        self.fused = fused
        if fused:
            self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.proj = nn.Linear(dim, hidden_dim, bias=False)
            self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        if self.fused: #untested
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            act = F.silu(gate).mul_(up)
            return self.out(act)
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


def _apply_rope1(x, freqs_cis):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])

    return x_out.reshape(*x.shape).type_as(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def forward(self, x, pe, prefix_len, negpip):
        # (B, N, H, D) throughout
        B, N, _ = x.shape

        q = self.to_q(x).view(B, N, self.heads, -1)
        q = self.norm_q(q)
        q = _apply_rope1(q, pe)

        k = self.to_k(x).view(B, N, self.heads, -1)
        k = self.norm_k(k)
        k = _apply_rope1(k, pe)

        v = self.to_v(x)
        if negpip is not None:
            y_len = len(negpip)
            v[:, :y_len, :] *= negpip[:, None]

        attn = attention_function(q.flatten(2), k.flatten(2), v, self.heads)
        return self.to_out[0](attn)


def _split_rows(p):
    # shared modulation rows: (t = 0 row for text and references, sampled-t rows for the target)
    return p[:1].unsqueeze(1), p[1:].unsqueeze(1)


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3, eps=1e-6, fused_mlp=True):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio, fused=fused_mlp)

    def forward(self, x, mod, pe, prefix_len, negpip):
        (s_prefix1, s_target1), (g_prefix1, g_target1), (s_prefix2, s_target2), (g_prefix2, g_target2) = mod
        norm = self.img_norm1(x)
        norm[:, :prefix_len].mul_(s_prefix1)
        norm[:, prefix_len:].mul_(s_target1)

        attn = self.attn(norm, pe, prefix_len, negpip)
        attn[:, :prefix_len].mul_(g_prefix1)
        attn[:, prefix_len:].mul_(g_target1)
        x.add_(attn)

        norm = self.img_norm2(x)
        norm[:, :prefix_len].mul_(s_prefix2)
        norm[:, prefix_len:].mul_(s_target2)

        mlp = self.img_mlp(norm)
        mlp[:, :prefix_len].mul_(g_prefix2)
        mlp[:, prefix_len:].mul_(g_target2)
        x.add_(mlp)

        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)

        return x


class LastLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        x = scale.add_(1) * self.norm(x)
        return x


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        fused_mlp=True,
        image_model=None,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        # one modulation shared by every block
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList([
            QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, fused_mlp=fused_mlp)
            for _ in range(num_layers)
        ])

        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

        # text + reference K/V are step-independent (t = 0 modulation, causal prefix), cached for one sampling run


    def build_sequence(self, x, context, ref_latents, ref_strengths):
        # text with each reference image spliced in at its slot, target image last
        txt = self.txt_in(context)
        parts, ids = [], []

        conds = (txt[:, 8:15], txt[:, 15:22], txt[:, 22:29], txt[:, 29:36], txt[:, -5:])

        parts.append(txt[:, 5:8])
        parts.append(txt[:, 36:-5])
        pos = txt.shape[1] - 38
        ids.append(torch.arange(0, pos, device=x.device, dtype=torch.float32).unsqueeze(1).expand(pos, 3))

        for (img, str, cond) in zip(ref_latents + [x], ref_strengths + [1.0], conds):
            if img is not None and str > 0.0:
                parts.append(cond)
                cond_len = cond.shape[1]
                ids.append(torch.arange(pos, pos+cond_len, device=x.device, dtype=torch.float32).unsqueeze(1).expand(cond_len, 3))
                pos += cond_len

                h, w = img.shape[-2:]
                parts.append(self.img_in(img.flatten(2).transpose(1, 2)).mul_(str)) # mul on img or result of img_in is identical

                hh = torch.arange(-(h - h // 2), h // 2, device=x.device, dtype=torch.float32)
                ww = torch.arange(-(w - w // 2), w // 2, device=x.device, dtype=torch.float32)

                ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
                pos += max(h, w)

        # (1, N, 1, ...): the layout the fused rms_rope wants for (B, N, H, D) queries
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe


    def forward(self, x, timesteps, context, negpip=None, **kwargs):
        B, C, H, W = x.shape
        dtype = x.dtype

        timestep = timesteps[0].item()
        ref_strengths = [s*timestep for s in getattr(shared, "klein_strength", (0.0, 0.0, 0.0, 0.0))]
        ref_latents = getattr(shared, "klein_latents", [None, None, None, None]) # lengths must match, currently 4 hardcoded

        hidden_states, pe = self.build_sequence(x, context, ref_latents, ref_strengths)
        prefix_len = hidden_states.shape[1] - H * W

        t = timesteps[:1].to(dtype)
        temb = self.time_text_embed(torch.cat((t.new_zeros(1), t)), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1.add_(1)), _split_rows(gate1.tanh()), _split_rows(scale2.add_(1)), _split_rows(gate2.tanh()))

        if negpip is not None:
            negpip = negpip[0]

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mod, pe, prefix_len, negpip)

        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[1:])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)

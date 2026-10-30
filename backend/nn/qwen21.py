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


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = nn.RMSNorm(in_dim, eps=eps, add=True)
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


_cache_type = "No cache"
_cache_method = "attn" #"attn", "kv", "attn+kv"
_cache_reset = False # flag set if context and/or references changed
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

        self.cached_k = None
        self.cached_v = None
        self.cached_attn = None

    def forward(self, x, pe, prefix_len, negpip, segments):
        # combined method gives most speedgain but uses most memory for cache
        # attn method uses less memory than kv method, closer to uncached output, maybe more gain
        global _cache_type, _cache_method, _cache_reset
        B, N, _ = x.shape

        q = self.to_q(x).view(B, N, self.heads, -1)
        q = self.norm_q(q)
        q = _apply_rope1(q, pe).flatten(2)
        
        no_kv_cache   = "kv"   not in _cache_method or _cache_type == "No cache"
        no_attn_cache = "attn" not in _cache_method or _cache_type == "No cache"

        if no_kv_cache or _cache_reset:
            k = self.to_k(x).view(B, N, self.heads, -1)
            k = self.norm_k(k)
            k = _apply_rope1(k, pe).flatten(2)

            v = self.to_v(x)

            if no_kv_cache:
                self.cached_k = None
                self.cached_v = None
            else:
                match _cache_type:
                    case "computation dtype":
                        self.cached_k = k[:, :prefix_len].clone()
                        self.cached_v = v[:, :prefix_len].clone()
                    case "fp16":
                        self.cached_k = k[:, :prefix_len].to(torch.float16)
                        self.cached_v = v[:, :prefix_len].to(torch.float16)
                    case "fp8":
                        self.cached_k = k[:, :prefix_len].to(torch.float8_e4m3fn)
                        self.cached_v = v[:, :prefix_len].to(torch.float8_e4m3fn)
                    case "computation dtype (CPU)":
                        self.cached_k = k[:, :prefix_len].clone().cpu()
                        self.cached_v = v[:, :prefix_len].clone().cpu()
                    case "fp16 (CPU)":
                        self.cached_k = k[:, :prefix_len].to(torch.float16).cpu()
                        self.cached_v = v[:, :prefix_len].to(torch.float16).cpu()
                    case "fp8 (CPU)":
                        self.cached_k = k[:, :prefix_len].to(torch.float8_e4m3fn).cpu()
                        self.cached_v = v[:, :prefix_len].to(torch.float8_e4m3fn).cpu()
                    case _:
                        self.cached_k = None
                        self.cached_v = None

        else:
            k = self.to_k(x[:, prefix_len:]).view(B, N-prefix_len, self.heads, -1)
            k = self.norm_k(k)
            k = _apply_rope1(k, pe[:, prefix_len:]).flatten(2)

            v = self.to_v(x[:, prefix_len:])

            k = torch.cat((self.cached_k.to(device=k.device).to(dtype=k.dtype), k), dim=1)
            v = torch.cat((self.cached_v.to(device=v.device).to(dtype=v.dtype), v), dim=1)

        if negpip is not None:
            v[:, :segments[0], :] *= negpip[:, None]

        attn = torch.empty_like(x)
        img_start = segments[-1] + 5
        if no_attn_cache or _cache_reset:
            start = 0
            for s in segments:
                attn[:, start:start+s] = attention_function(q[:, start:start+s], k[:, :start+s], v[:, :start+s], self.heads)
                start += s

            if no_attn_cache:
                self.cached_attn = None
            else:
                match _cache_type:
                    case "computation dtype":
                        self.cached_attn = attn[:, :-img_start].clone()
                    case "fp16":
                        self.cached_attn = attn[:, :-img_start].to(torch.float16)
                    case "fp8":
                        self.cached_attn = attn[:, :-img_start].to(torch.float8_e4m3fn)
                    case "computation dtype (CPU)":
                        self.cached_attn = attn[:, :-img_start].clone().cpu()
                    case "fp16 (CPU)":
                        self.cached_attn = attn[:, :-img_start].to(torch.float16).cpu()
                    case "fp8 (CPU)":
                        self.cached_attn = attn[:, :-img_start].to(torch.float8_e4m3fn).cpu()
                    case _:
                        self.cached_attn = None

        else:
            attn[:, :-img_start] = self.cached_attn.to(device=x.device).to(dtype=x.dtype)
            attn[:, -img_start:] = attention_function(q[:, -img_start:], k, v, self.heads)

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

    def forward(self, x, mod, pe, prefix_len, negpip, segments):
        (s_prefix1, s_target1), (g_prefix1, g_target1), (s_prefix2, s_target2), (g_prefix2, g_target2) = mod
        norm = self.img_norm1(x)
        norm[:, :prefix_len].mul_(s_prefix1)
        norm[:, prefix_len:].mul_(s_target1)

        attn = self.attn(norm, pe, prefix_len, negpip, segments)
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
            x = x.clamp_(-65504, 65504)

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
        inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(inner_dim)
        self.txt_in = TextProjection(context_in_dim, inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, inner_dim, bias=False)

        # one modulation shared by every block
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 4 * inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList([
            QwenImage21TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, fused_mlp=fused_mlp)
            for _ in range(num_layers)
        ])

        self.norm_out = LastLayer(inner_dim, eps=eps)
        self.proj_out = nn.Linear(inner_dim, out_channels, bias=False)

        self.last_context_hash = 0
        self.last_reference_hash = 0
        self.last_reference_strengths = (0.0, 0.0, 0.0, 0.0)

        # text + reference K/V are step-independent (t = 0 modulation, causal prefix), cached for one sampling run


    def build_sequence(self, x, context, ref_latents, ref_strengths):
        b = x.shape[0]
        # text with each reference image spliced in at its slot, target image last
        txt = self.txt_in(context)
        parts, ids, segments = [], [], []

        # if using full system prompt
        # conds = (txt[:, 17:24], txt[:, 24:31], txt[:, 31:38], txt[:, 38:45], txt[:, -5:])

        # parts.append(txt[:, 14:17])
        # parts.append(txt[:, 45:-5])
        # pos = txt.shape[1] - 47

        conds = (txt[:, 8:15], txt[:, 15:22], txt[:, 22:29], txt[:, 29:36], txt[:, -5:])

        parts.append(txt[:, 5:8])
        parts.append(txt[:, 36:-5])
        pos = txt.shape[1] - 38
        segments.append(pos)

        ids.append(torch.arange(0, pos, device=x.device, dtype=torch.float32).unsqueeze(1).expand(pos, 3))

        for (img, str, cond) in zip(ref_latents + [x], ref_strengths + (1.0,), conds):
            if img is not None and str > 0.0:
                parts.append(cond)
                cond_len = cond.shape[1]
                ids.append(torch.arange(pos, pos+cond_len, device=x.device, dtype=torch.float32).unsqueeze(1).expand(cond_len, 3))
                pos += cond_len

                h, w = img.shape[-2:]
                p_img = self.img_in(img.flatten(2).transpose(1, 2)).mul_(str)
                if b > 1: # batch
                    p_img = p_img.expand(b, *p_img.shape[1:])
                parts.append(p_img)

                hh = torch.arange(-(h - h // 2), h // 2, device=x.device, dtype=torch.float32)
                ww = torch.arange(-(w - w // 2), w // 2, device=x.device, dtype=torch.float32)

                ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
                pos += max(h, w)
                segments.append(cond_len + h*w)

        # (1, N, 1, ...): the layout the fused rms_rope wants for (B, N, H, D) queries
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments


    def forward(self, x, timesteps, context, negpip=None, **kwargs):
        global _cache_type, _cache_reset
        B, C, H, W = x.shape
        dtype = x.dtype

        context_hash = hash(context.cpu().numpy().tobytes())
        if context_hash != self.last_context_hash:
            _cache_reset = True
            self.last_context_hash = context_hash

        _cache_hash = getattr(shared, "klein_image_hash", 0)
        if _cache_hash != self.last_reference_hash:
            _cache_reset = True
            self.last_reference_hash = _cache_hash

        ref_strengths = getattr(shared, "klein_strength", (0.0, 0.0, 0.0, 0.0))
        if ref_strengths != self.last_reference_strengths:
            _cache_reset = True
            self.last_reference_strengths = ref_strengths

        ref_latents = getattr(shared, "klein_latents", [None, None, None, None]) # lengths must match, currently 4 hardcoded
        _cache_type = getattr(shared.opts, "qwen2.1_kv_cache", "No cache")

        if _cache_type == "No cache":
            self.last_reference_hash = 0

        hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, ref_strengths)
        prefix_len = hidden_states.shape[1] - H * W

        t = timesteps[:1].to(dtype)
        temb = self.time_text_embed(torch.cat((t.new_zeros(1), t)), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1.add_(1)), _split_rows(gate1.tanh()), _split_rows(scale2.add_(1)), _split_rows(gate2.tanh()))

        if negpip is not None:
            negpip = negpip[0]

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mod, pe, prefix_len, negpip, segments)

        _cache_reset = False

        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[1:])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)

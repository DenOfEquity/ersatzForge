# https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.attention import attention_function

from backend.nn.flux import EmbedND
from backend.nn.mmditx import TimestepEmbedder


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


class JointAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        t_ = x_in.reshape(*x_in.shape[:-1], -1, 1, 2)
        t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
        return t_out.reshape(*x_in.shape)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = JointAttention.apply_rotary_emb(xq, freqs_cis=freqs_cis)

        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xk = self.k_norm(xk)
        xk = JointAttention.apply_rotary_emb(xk, freqs_cis=freqs_cis)

        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = attention_function(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, None, skip_reshape=True)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float, qk_norm: bool, modulation=True, z_modulation=False):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.modulation = modulation
        if modulation:
            if z_modulation:
                self.adaLN_modulation = nn.Sequential(
                    nn.Linear(min(dim, 256), 4 * dim, bias=True),
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(min(dim, 1024), 4 * dim, bias=True),
                )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, adaln_input: torch.Tensor = None):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x.add_(gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                )))
            x.add_(gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )))
        else:
            assert adaln_input is None
            x.add_(self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                )))
            x.add_(self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )))
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels, z_modulation=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 256 if z_modulation else 1024), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class Lumina2DiT(nn.Module):

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: int = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 4.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (1, 512, 512),
        rope_theta: float = 10000.0,
        z_modulation: bool = False,
        pad_tokens_multiple = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = nn.Linear(in_features=patch_size * patch_size * in_channels, out_features=dim, bias=True)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_modulation=z_modulation,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=256 if z_modulation else None)
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    z_modulation=z_modulation,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.norm_final = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, z_modulation=z_modulation)

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(theta=rope_theta, axes_dim=axes_dims)
        self.dim = dim
        self.n_heads = n_heads

    def unpatchify(self, x: torch.Tensor, img_size: tuple[int, int], cap_size: int) -> list[torch.Tensor]:
        pH = pW = self.patch_size
        H, W = img_size

        begin = cap_size
        end = begin + (H // pH) * (W // pW)

        x = x[:, begin:end]
        x = rearrange(x, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h=H // pH, w=W // pW, ph=pH, pw=pW)

        return x

    def patchify_and_embed(self, x: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor, num_tokens) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[int], torch.Tensor]:
        bsz = x.shape[0]
        pH = pW = self.patch_size
        device = x.device
        dtype = x.dtype

        if cap_mask is not None:
            num_tokens = cap_mask.sum(dim=1).item()
            if not torch.is_floating_point(cap_mask):
                cap_mask = (cap_mask - 1).to(dtype) * torch.finfo(dtype).max

        img_size = [x.shape[2], x.shape[3]]
        img_len = (x.shape[2] // pH) * (x.shape[3] // pW)

        max_seq_len = num_tokens + img_len
        if self.pad_tokens_multiple is not None:
            pad_extra = (-max_seq_len) % self.pad_tokens_multiple
            max_seq_len += pad_extra
            
        H, W = img_size
        H_tokens, W_tokens = H // pH, W // pW
        row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)
        for i in range(bsz):
            position_ids[i, :num_tokens, 0] = torch.arange(num_tokens, dtype=torch.int32, device=device)
            position_ids[i, num_tokens : num_tokens + img_len, 0] = num_tokens
            position_ids[i, num_tokens : num_tokens + img_len, 1] = row_ids
            position_ids[i, num_tokens : num_tokens + img_len, 2] = col_ids

        x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=pH, pw=pW)

        if self.pad_tokens_multiple is not None:
            pad_extra = (-img_len) % self.pad_tokens_multiple
        else:
            pad_extra = 0

        padded_img_embed = torch.zeros(bsz, img_len+pad_extra, x.shape[-1], device=device, dtype=dtype)
        padded_img_embed[:, :img_len] = x
        padded_img_embed = self.x_embedder(padded_img_embed)

        freqs_cis = self.rope_embedder(position_ids).movedim(1, 2).to(dtype)

        cap_freqs_cis_shape = list(freqs_cis.shape)
        cap_freqs_cis_shape[1] = cap_feats.shape[1]
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = img_len+pad_extra
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        for i in range(bsz):
            cap_freqs_cis[i, :num_tokens] = freqs_cis[i, :num_tokens]
            img_freqs_cis[i, :img_len] = freqs_cis[i, num_tokens : num_tokens + img_len]

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)


        padded_img_mask = torch.zeros(bsz, img_len+pad_extra, dtype=dtype, device=device)
        for i in range(bsz):
            padded_img_mask[i, img_len :] = -torch.finfo(dtype).max
        padded_img_mask = padded_img_mask.unsqueeze(1)

        for layer in self.noise_refiner:
            padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t)

        if cap_mask is not None:
            mask = torch.zeros(bsz, max_seq_len, dtype=dtype, device=device)
            mask[:, :num_tokens] = cap_mask[:, :num_tokens]
        else:
            mask = None

        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=dtype)
        for i in range(bsz):
            padded_full_embed[i, :num_tokens] = cap_feats[i, :num_tokens]
            padded_full_embed[i, num_tokens : num_tokens + img_len] = padded_img_embed[i, :img_len]

        return padded_full_embed, mask, img_size, freqs_cis

    def forward(self, x, timesteps, context, num_tokens=None, attention_mask=None, **kwargs):
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape

        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

        if self.dim == 2304: # Lumina 2
            t = self.t_embedder(t, dtype=x.dtype)
        else: # Z image
            t = self.t_embedder(t*1000.0, dtype=x.dtype)
        adaln_input = t

        if self.pad_tokens_multiple is not None:
            pad_t = (-context.shape[1]) % self.pad_tokens_multiple
            pad = context.new_zeros([bs, pad_t, context.shape[2]])
            context = torch.cat([context, pad], dim=1)
        cap_feats = self.cap_embedder(context)

        # entire batch will have same length context because of calc_cond_uncond_batch
        # (it seems the Lumina processing originally handled different lengths, but I've simplified it out as that'll never happen in this webUI)
        num_tokens = context.shape[1] # doesn't account for padding, but none is used
        x, mask, img_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t, num_tokens=num_tokens)
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, num_tokens)[:, :, :h, :w]

        return -x

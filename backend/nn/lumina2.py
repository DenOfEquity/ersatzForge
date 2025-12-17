# https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.attention import attention_function

from backend.nn.flux import EmbedND, FluxPosEmbed
from backend.nn.mmditx import TimestepEmbedder

from modules import shared


# fp16 fix overflow by downscaling github.com/comfyanonymous/ComfyUI/pull/11187 by vanDuven


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
        # t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
        t_out = torch.mul(freqs_cis[..., 0], t_[..., 0])
        t_out.addcmul_(freqs_cis[..., 1], t_[..., 1])
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
        xq = JointAttention.apply_rotary_emb(xq, freqs_cis)

        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xk = self.k_norm(xk)
        xk = JointAttention.apply_rotary_emb(xk, freqs_cis)

        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = attention_function(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, None, skip_reshape=True)

        if output.dtype == torch.float16:
            output.div_(4)

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
        if x.dtype == torch.float16:
            return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x).div_(32)))
        else:
            return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float, qk_norm: bool, modulation=True, z_modulation=False, block_id=None):
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
        self.block_id = block_id
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

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, adaln_input: torch.Tensor=None, control=None, strength=None):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x.add_(gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    self.attention_norm1(x).mul_(scale_msa.add_(1.0).unsqueeze(1)),
                    x_mask,
                    freqs_cis,
                )))
            x.add_(gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x).mul_(scale_mlp.add_(1.0).unsqueeze(1)),
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

        if self.block_id is not None and control is not None and strength is not None:
            control_size = control[self.block_id].shape[1]
            x[:, -control_size:] += control[self.block_id] * strength

        return x


class JointTransformerBlockControl(JointTransformerBlock):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float, qk_norm: bool, modulation=True, z_modulation=False, block_id=None, control=False):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm, modulation, z_modulation)
        self.block_id = block_id
        if control and block_id is not None:
            if block_id == 0:
                self.before_proj = nn.Linear(self.dim, self.dim)
            self.after_proj = nn.Linear(self.dim, self.dim)


    def forward(self, c: torch.Tensor, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, adaln_input: torch.Tensor=None):
        if self.block_id is not None:
            if self.block_id == 0:
                c = self.before_proj(c) + x
                all_c = []
            else:
                all_c = list(torch.unbind(c))
                c = all_c.pop(-1)

        c = super().forward(c, x_mask, freqs_cis, adaln_input)
        
        if self.block_id is not None:
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)

        return c


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
        x = self.norm_final(x).mul_(scale.add_(1).unsqueeze(1))
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
        num_keys: int = None,
        Z_image_control_2_0_broken = False,
    ):
        super().__init__()
        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.use_dynamicPE = shared.opts.dynamicPE_lumina2
        if self.use_dynamicPE > 0:
            self.rope_embedder = FluxPosEmbed(theta=rope_theta, axes_dim=axes_dims, base_resolution=self.use_dynamicPE)
        else:
            self.rope_embedder = EmbedND(theta=rope_theta, axes_dim=axes_dims)

        self.dim = dim
        self.n_heads = n_heads

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = nn.Linear(in_features=patch_size * patch_size * in_channels, out_features=dim, bias=True)

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
        
        #400 keys: lumina2, 455 keys: ZITurbo (maybe 1-3 less if pad tokens or sigmas not included)
        self.add_control_noise_refiner = False
        self.add_control_noise_refiner_correct = Z_image_control_2_0_broken
        if num_keys > 455:  # z-image-turbo + control
            if num_keys > 575:  #version 2
                layers = 15
                control_layers_places = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
                control_refiner_places = [0, 1]
                self.control_x_embedder = nn.Linear(1 * 2 * 2 * 33, dim, bias=True)
                self.add_control_noise_refiner = True
            else:
                layers = 6
                control_layers_places = [0, 5, 10, 15, 20, 25]
                control_refiner_places = []
                self.control_x_embedder = nn.Linear(1 * 2 * 2 * 16, dim, bias=True)

            self.control_layers = nn.ModuleList(
                [
                    JointTransformerBlockControl(
                        i, 
                        dim, 
                        n_heads, 
                        n_kv_heads, 
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps, 
                        qk_norm,
                        z_modulation=z_modulation,
                        control=True,
                        block_id=i,
                    )
                    for i in control_layers_places
                ]
            )

            self.control_noise_refiner = nn.ModuleList(
                [
                    JointTransformerBlockControl(
                        layer_id + 1000,
                        dim,
                        n_heads,
                        n_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        z_modulation=z_modulation,
                        control=True if self.add_control_noise_refiner else False,
                        block_id=layer_id if self.add_control_noise_refiner else None,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
            control_layers_mapping = {i: n for n, i in enumerate(control_layers_places)}
            control_refiner_mapping = {i: n for n, i in enumerate(control_refiner_places)}
            self.control = True
        else: #controlnet not added
            control_layers_places = []
            control_layers_mapping = {}
            control_refiner_places = []
            control_refiner_mapping = ()
            self.control = False

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id + 1000,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_modulation=z_modulation,
                    block_id=control_refiner_mapping[layer_id] if layer_id in control_refiner_places else None,
                )
                for layer_id in range(n_refiner_layers)
            ]
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
                    block_id=control_layers_mapping[layer_id] if layer_id in control_layers_places else None,
                )
                for layer_id in range(n_layers)
            ]
        )

        self.norm_final = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, z_modulation=z_modulation)

        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))


    def unpatchify(self, x: torch.Tensor, img_size: tuple[int, int], cap_size: int) -> list[torch.Tensor]:
        pH = pW = self.patch_size
        H, W = img_size

        begin = cap_size
        end = begin + (H // pH) * (W // pW)

        x = x[:, begin:end]
        x = rearrange(x, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h=H // pH, w=W // pW, ph=pH, pw=pW)

        return x

    def patchify_and_embed(self, x: torch.Tensor, context: torch.Tensor, control: torch.Tensor, t: torch.Tensor, num_tokens, timestep=None) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[int], torch.Tensor]:
        bsz = x.shape[0]
        pH = pW = self.patch_size
        device = x.device
        dtype = x.dtype

        H, W = x.shape[2], x.shape[3]

        cap_pos_ids = torch.zeros(bsz, num_tokens, 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(num_tokens, dtype=torch.float32, device=device) + 1.0

        x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=pH, pw=pW)
        x = self.x_embedder(x)

        H_tokens, W_tokens = H // pH, W // pW
        row_ids = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        col_ids = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
        # if not self.use_dynamicPE and shared.opts.rope_scaling:
            # h_scale = 1.0
            # w_scale = 1.0
            # limit = 2048 // 16
            # if H_tokens > limit:
                # h_scale = limit / H_tokens
            # if W_tokens > limit:
                # w_scale = limit / W_tokens
            # row_ids *= min(h_scale, w_scale)
            # col_ids *= min(h_scale, w_scale)

        position_ids = torch.zeros(bsz, x.shape[1], 3, dtype=torch.float32, device=device)
        position_ids[:,:, 0] = num_tokens + 1
        position_ids[:,:, 1] = row_ids
        position_ids[:,:, 2] = col_ids

        if control is not None:
            control = self.control_x_embedder(control)

        if self.pad_tokens_multiple is not None:
            pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
            if pad_extra:
                pad = self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)
                x = torch.cat((x, pad), dim=1)
                if control is not None:
                    control = torch.cat((control, pad), dim=1)
                position_ids = torch.nn.functional.pad(position_ids, (0, 0, 0, pad_extra))

        ids = torch.cat((cap_pos_ids, position_ids), dim=1)
        if self.use_dynamicPE:
            self.rope_embedder.set_timestep(timestep.item())
            pes = []
            for i in range(ids.shape[0]):
                pe = self.rope_embedder(ids[i])

                out = torch.stack([pe[0], -pe[1], pe[1], pe[0]], dim=-1).unsqueeze(0)
                out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
                pes.append(out.unsqueeze(1))
            freqs_cis = torch.cat(pes, dim=0)
        else:
            freqs_cis = self.rope_embedder(ids)
        freqs_cis = freqs_cis.movedim(1, 2).to(dtype)

        hints = None
        if control is not None:
            if self.add_control_noise_refiner and self.add_control_noise_refiner_correct: # v2.1
                hints = control
                for layer in self.control_noise_refiner:
                    hints = layer(hints, x, None, freqs_cis[:, num_tokens:], t)
                hints = torch.unbind(hints)[:-1]
               
            if not self.add_control_noise_refiner:
                for layer in self.control_noise_refiner:
                    control = layer(control, x, None, freqs_cis[:, num_tokens:], t)

            for layer in self.control_layers:
                control = layer(control, x, None, freqs_cis[:, num_tokens:], t)
            control = torch.unbind(control)[:-1]

        for layer in self.noise_refiner:
            x = layer(x, None, freqs_cis[:, num_tokens:], t, hints or control, strength=2.0)

        for layer in self.context_refiner:
            context = layer(context, None, freqs_cis[:, :num_tokens])

        padded_full_embed = torch.cat((context, x), dim=1)

        return padded_full_embed, control, freqs_cis

    def forward(self, x, timesteps, context, num_tokens=None, attention_mask=None, **kwargs):
        t = 1.0 - timesteps
        bs, c, h, w = x.shape

        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

        if self.control:
            control = getattr(shared, 'ZITlatent', None) # already patchified
            control_strength = getattr(shared, 'ZITstrength', 0.0)
            control_stop_sigma = getattr(shared, 'ZITstop', 0.0)
            if control_strength == 0.0:
                control = None
            elif timesteps[0] < control_stop_sigma:
                control = None
            if control is not None and bs > 1:
                control = control.repeat(bs, 1, 1)
        else:
            control = None
            control_strength = 0.0

        if self.dim == 2304: # Lumina 2
            adaln_input = self.t_embedder(t, dtype=x.dtype)
        else: # Z image
            adaln_input = self.t_embedder(t*1000.0, dtype=x.dtype)

        #cache this?
        if self.pad_tokens_multiple is not None:
            pad_t = (-context.shape[1]) % self.pad_tokens_multiple
            pad = context.new_zeros([bs, pad_t, context.shape[2]])
            context = torch.cat([context, pad], dim=1)
        context = self.cap_embedder(context)
        #pad using cap_pad_token here? #self.cap_pad_token.repeat(bs, pad_t, 1).to(cap_feats)

        # entire batch will have same length context because of calc_cond_uncond_batch
        # (it seems the Lumina processing originally handled different lengths, but I've simplified it out as that'll never happen in this webUI)
        num_tokens = context.shape[1] # doesn't account for padding
        x, control, freqs_cis = self.patchify_and_embed(x, context, control, adaln_input, num_tokens=num_tokens, timestep=t)
        freqs_cis = freqs_cis.to(x.device)


        for layer in self.layers:
            x = layer(x, None, freqs_cis, adaln_input, control, control_strength)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, [h, w], num_tokens)[:, :, :h, :w]

        return -x

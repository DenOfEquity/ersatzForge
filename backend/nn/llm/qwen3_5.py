from dataclasses import dataclass, field
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.attention import attention_pytorch

from .llama import BaseLlama, MLP, apply_rope, precompute_freqs_cis


def _qwen35_layer_types(n):
    return [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(n)]


@dataclass
class Qwen35_9BConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    mrope_section: list = field(default_factory=lambda: [11, 11, 10])
    layer_types: list = field(default_factory=lambda: _qwen35_layer_types(32))
    rms_norm_add: bool = True
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    final_norm: bool = True
    lm_head: bool = False
    transformer_type: str = "qwen35_9b"
    head_dim = 256
    rope_dims = None
    rope_scale = None


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False):
    initial_dtype = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key   = F.pad(key,   (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta  = F.pad(beta,  (0, pad_size))
    g     = F.pad(g,     (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key   * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35_9BConfig):
        super().__init__()

        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(value_dim, hidden, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads), requires_grad=False)
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads), requires_grad=False)

        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=conv_dim, padding=self.conv_kernel_size - 1)

        self.norm = nn.RMSNorm(self.value_head_dim, eps=config.rms_norm_eps)

    def forward(self, x, **kwargs):
        batch_size, seq_len, _ = x.shape

        proj = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        mixed_qkv = proj.transpose(1, 2)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        g_decay = -self.A_log.to(x.device, torch.float32).exp()
        dt_bias = self.dt_bias.to(x.device, torch.float32)

        beta = b.sigmoid()
        g = g_decay * F.softplus(a.float() + dt_bias)
        query = query.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core_attn_out, _ = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False)

        core_attn_out = self.norm(core_attn_out.reshape(-1, self.value_head_dim)) * F.silu(z.reshape(-1, self.value_head_dim))
        output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
        return output


def apply_partial_rope(xq, xk, freqs_cis, rotary_dim):
    xq_rot = xq[..., :rotary_dim]
    xq_pass = xq[..., rotary_dim:]
    xk_rot = xk[..., :rotary_dim]
    xk_pass = xk[..., rotary_dim:]

    xq_rot, xk_rot = apply_rope(xq_rot, xk_rot, freqs_cis)

    xq = torch.cat([xq_rot, xq_pass], dim=-1)
    xk = torch.cat([xk_rot, xk_pass], dim=-1)
    return xq, xk


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35_9BConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.inner_size = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size * 2, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add)

    def forward(self, x, attention_mask=None, freqs_cis=None, **kwargs):
        batch_size, seq_length, _ = x.shape

        qg = self.q_proj(x)
        qg = qg.view(batch_size, seq_length, self.num_heads, self.head_dim * 2)
        xq, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        gate = gate.reshape(batch_size, seq_length, -1)

        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_partial_rope(xq, xk, freqs_cis, self.rotary_dim)

        gqa_kwargs = {"enable_gqa": True} if self.num_heads != self.num_kv_heads else {}
        output = attention_pytorch(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, **gqa_kwargs)

        output = output * gate.sigmoid()
        return self.o_proj(output)


class Qwen35TransformerBlock(nn.Module):
    def __init__(self, config: Qwen35_9BConfig, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)

    def forward(self, x, attention_mask=None, freqs_cis=None, **kwargs):
        if self.layer_type == "linear_attention":
            x += self.linear_attn(self.input_layernorm(x), attention_mask=attention_mask)
        else:
            x += self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, freqs_cis=freqs_cis)

        x += self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen35Transformer(nn.Module):
    def __init__(self, config: Qwen35_9BConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen35TransformerBlock(config, index=i) for i in range(config.num_hidden_layers)])

        if config.final_norm:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        else:
            self.norm = None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[], **kwargs):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x).to(dtype=dtype)

        seq_len = x.shape[1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        freqs_cis = precompute_freqs_cis(rotary_dim, position_ids, self.config.rope_theta, self.config.rope_scale, self.config.mrope_section, interleaved_mrope=True, device=x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask


        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            x = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis)

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if all_intermediate is not None:
            all_intermediate.append(x.unsqueeze(1).clone())
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            intermediate = self.norm(intermediate)

        return x, intermediate


class Qwen35TextModel(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Qwen35_9BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers
        self.model = Qwen35Transformer(config)


import torch
import torch.nn as nn
from einops import rearrange

# from backend.attention import attention_function


def time_embedding(sigma: torch.Tensor, dim: int = 256) -> torch.Tensor:
    # upstream's timesteps are the scheduler's float32 sigmas times 1000; the sinusoid is computed in float64 and fed
    # in float32
    times = (sigma.float() * 1000.0).double()
    half = dim // 2
    freq = 10000 ** (-torch.arange(half, device=sigma.device, dtype=torch.float64) / half)
    phase = times[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), dim=-1).float()


def rope_table(h: int, w: int, text_len: int, head_dim: int, base: float, device) -> torch.Tensor:
    """(H*W + T, head_dim): sin then cos of the four axes' phases, image tokens first, in float32."""
    num_bands = head_dim // 8
    bands = 1 / (base ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))
    bands = bands.to(device)

    ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
    ids[: h * w, 1] = text_len
    ids[: h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
    ids[: h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
    ids[h * w :, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    phase = ids.unsqueeze(-1) * bands

    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x (B, L, H, D), rope (L, D): rotation by halves, in x's dtype (upstream casts the table to the activations')
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1)[:, None]
    cos = torch.cat((cos, cos), -1)[:, None]
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope, heads):
    """q, k, v (B, L, H*D) -> (B, L, H*D), RoPE on q and k, every token attending to every other."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)
    #return attention_function(q.to(v.dtype), k.to(v.dtype), v, q.shape[1], skip_reshape=True)


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


class TextBlock(nn.Module):
    """Text refinement: pre-norm attention and SiTU MLP, no modulation, biases on every projection."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.q_norm = nn.RMSNorm(dim // heads)
        self.k_norm = nn.RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, context, rope):
        x = self.proj(context)
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, fused_qkv=False):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.q_norm = nn.RMSNorm(dim // heads)
        self.k_norm = nn.RMSNorm(dim // heads)

        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({n: nn.Linear(dim, 3 * dim if n == "qkv" else dim, bias=False) for n in names})

        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, mod, negpip=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod

        y = torch.addcmul(shift_a, scale_a, self.norm1(x))

        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, -1)
        else:
            q = self.proj["q"](y)
            k = self.proj["k"](y)
            v = self.proj["v"](y)

        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)

        if negpip is not None:
            txtlen = len(negpip)
            v[:, -txtlen:, :] *= negpip[:, None]

        out = self.proj["o"](attention(q, k, v, rope, self.heads))
        x.addcmul_(out, gate_a)

        y = torch.addcmul(shift_f, scale_f, self.norm2(x))

        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        x.addcmul_(self.proj["ffn_down"](situ(hidden, gate)), gate_f)
        return x


class MossImageDiT(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                 latent_channels=128, rope_base=10000.0, fused_qkv=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.rope_base = rope_base
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv) for _ in range(num_layers)])
        self.final_norm_video = nn.RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(self, x, timestep, context, negpip=None, transformer_options=None, **kwargs):
        """x (B, 128, H, W) latents, timestep (B,) = sigma, context (B, T, 4096) -> velocity (B, 128, H, W)."""
        b, _, h, w = x.shape
        h //= 2
        w //= 2
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        device = x.device
        dtype = x.dtype

        # context = context.to(device=device, dtype=dtype)

        time = self.time_embedding(time_embedding(timestep, self.freq_dim).to(dtype))
        mod = self.shared_block_modulation(time).view(b, 6, -1).to(dtype)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in mod.unbind(1))
        scale_a.add_(1)
        scale_f.add_(1)
        mod = (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f)

        head = self.video_head_modulation(time).view(b, 2, -1).to(dtype)

        image_tokens = h * w
        rope = rope_table(h, w, context.shape[1], self.dim // self.num_heads, self.rope_base, device).to(dtype)
        video = self.video_embedder(x)
        text = self.text_embedder(context, rope[image_tokens:].to(context))
        tokens = torch.cat((video, text), 1)
        del video, text

        if negpip is not None:
            negpip = negpip[0]

        for layer in self.layers:
            tokens = layer(tokens, rope, mod, negpip)

        out = self.final_norm_video(tokens[:, :image_tokens]) * (1 + head[:, 1, None]) + head[:, 0, None]
        out = self.final_linear_video(out)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=h, w=w)

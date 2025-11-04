# based on the ChromaDCT(Radiance) implementation in https://github.com/maybleMyers/chromaforge

import torch
from torch import Tensor, nn

from einops import rearrange
from functools import lru_cache

from .flux import RMSNorm, EmbedND, timestep_embedding, FluxPosEmbed
from .chroma import Approximator, DoubleStreamBlock, SingleStreamBlock

from modules import shared


class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).

    This module takes an input tensor of shape (B, P^2, C), where P is the
    patch size, and enriches it with positional information before projecting
    it to a new hidden size.
    """
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        """
        Initializes the NerfEmbedder.

        Args:
            in_channels (int): The number of channels in the input tensor.
            hidden_size_input (int): The desired dimension of the output embedding.
            max_freqs (int): The number of frequency components to use for both
                             the x and y dimensions of the positional encoding.
                             The total number of positional features will be max_freqs^2.
        """
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input

        # A linear layer to project the concatenated input features and
        # positional encodings to the final output dimension.
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input))

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size, device, dtype):
        """
        Generates and caches 2D DCT-like positional embeddings for a given patch size.

        The LRU cache is a performance optimization that avoids recomputing the
        same positional grid on every forward pass.

        Args:
            patch_size (int): The side length of the square input patch.
            device: The torch device to create the tensors on.
            dtype: The torch dtype for the tensors.

        Returns:
            A tensor of shape (1, patch_size^2, max_freqs^2) containing the
            positional embeddings.
        """
        # Create normalized 1D coordinate grids from 0 to 1.
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)

        # Create a 2D meshgrid of coordinates.
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        # Reshape positions to be broadcastable with frequencies.
        # Shape becomes (patch_size^2, 1, 1).
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        # Create a 1D tensor of frequency values from 0 to max_freqs-1.
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)

        # Reshape frequencies to be broadcastable for creating 2D basis functions.
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]

        # A custom weighting coefficient, not part of standard DCT.
        # This seems to down-weight the contribution of higher-frequency interactions.
        coeffs = (1 + freqs_x * freqs_y) ** -1

        # Calculate the 1D cosine basis functions for x and y coordinates.
        # This is the core of the DCT formulation.
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)

        # Combine the 1D basis functions to create 2D basis functions by element-wise
        # multiplication, and apply the custom coefficients. Broadcasting handles the
        # combination of all (pos_x, freqs_x) with all (pos_y, freqs_y).
        # The result is flattened into a feature vector for each position.
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)

        return dct

    def forward(self, inputs):
        """
        Forward pass for the embedder.

        Args:
            inputs (Tensor): The input tensor of shape (B, P^2, C).

        Returns:
            Tensor: The output tensor of shape (B, P^2, hidden_size_input).
        """
        # Get the batch size, number of pixels, and number of channels.
        B, P2, C = inputs.shape

        # Infer the patch side length from the number of pixels (P^2).
        patch_size = int(P2 ** 0.5)

        # Fetch the pre-computed or cached positional embeddings.
        dct = self.fetch_pos(patch_size, inputs.device, inputs.dtype)

        # Repeat the positional embeddings for each item in the batch.
        dct = dct.repeat(B, 1, 1)

        # Concatenate the original input features with the positional embeddings
        # along the feature dimension.
        inputs = torch.cat([inputs, dct], dim=-1)

        # Project the combined tensor to the target hidden size.
        inputs = self.embedder(inputs)

        return inputs


class NerfGLUBlock(nn.Module):
    """
    A NerfBlock using a Gated Linear Unit (GLU) like MLP.
    """
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio):
        super().__init__()
        # The total number of parameters for the MLP is increased to accommodate
        # the gate, value, and output projection matrices.
        # We now need to generate parameters for 3 matrices.
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = nn.Linear(hidden_size_s, total_params)
        self.norm = RMSNorm(hidden_size_x)
        self.mlp_ratio = mlp_ratio
        # nn.init.zeros_(self.param_generator.weight)
        # nn.init.zeros_(self.param_generator.bias)


    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params = self.param_generator(s)

        # Split the generated parameters into three parts for the gate, value, and output projection.
        fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)

        # Reshape the parameters into matrices for batch matrix multiplication.
        fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2_params.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        # Normalize the generated weight matrices as in the original implementation.
        fc1_gate = torch.nn.functional.normalize(fc1_gate, dim=-2)
        fc1_value = torch.nn.functional.normalize(fc1_value, dim=-2)
        fc2 = torch.nn.functional.normalize(fc2, dim=-2)

#        res_x = x
#        x = self.norm(x)

        # Apply the final output projection.
#        x = torch.bmm(torch.nn.functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)

#        x.add_(res_x)
#        return x

        n_x = self.norm(x)
        # Apply the final output projection.
        n_x = torch.bmm(torch.nn.functional.silu(torch.bmm(n_x, fc1_gate)) * torch.bmm(n_x, fc1_value), fc2)

        x.add_(n_x)
        return x


class NerfFinalLayerConv(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)

        # replace nn.Linear with nn.Conv2d since linear is just pointwise conv
        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x shape: [N, C, H, W] - RMSNorm normalizes over the last dimension, hence permute
        x_norm = self.norm(x.permute(0, 2, 3, 1))    # Apply normalization on the feature/channel dimension

        # Permute back to the original dimension order for the convolution
        x = self.conv(x_norm.permute(0, 3, 1, 2))    # Apply the 3x3 convolution
        return x


class IntegratedChromaDCTTransformer2DModel(nn.Module):
    def __init__(self, **config):
        super().__init__()

        # just hardcode these params
        self.in_channels = 3
        self.out_channels = 3
        self.patch_size = 16

        self.hidden_size = 3072
        self.num_heads = 24
        if shared.opts.use_dynamicPE:
            self.pe_embedder = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56], base_resolution=shared.opts.dynamicPE_base)
            self.use_dynamicPE = True
        else:
            self.pe_embedder = EmbedND(theta=10000, axes_dim=[16, 56, 56])
            self.use_dynamicPE = False

        # patchify ops
        self.img_in_patch = nn.Conv2d(3, 3072, kernel_size=16, stride=16, bias=True)
        nn.init.zeros_(self.img_in_patch.weight)
        nn.init.zeros_(self.img_in_patch.bias)

        # currently the mapping is hardcoded in distribute_modulations function
        self.distilled_guidance_layer = Approximator(64, 3072, 5120, 5)
        self.txt_in = nn.Linear(4096, 3072)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(3072, 24, mlp_ratio=4, qkv_bias=True)
                for _ in range(19)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(3072, 24, mlp_ratio=4)
                for _ in range(38)
            ]
        )

        # pixel channel concat with DCT
        self.nerf_image_embedder = NerfEmbedder(in_channels=3, hidden_size_input=64, max_freqs=8)

        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(hidden_size_s=3072, hidden_size_x=64, mlp_ratio=4) for _ in range(4)
        ])

        self.nerf_final_layer_conv = NerfFinalLayerConv(64, out_channels=3)

        self.mod_index_length = (3 * 38) * (6 * 19)
        self.depth_single_blocks = 38
        self.depth_double_blocks = 19

        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length))),
            persistent=False,
        )
        self.approximator_in_dim = 64


    def inner_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timestep: Tensor,
        guidance: Tensor,
    ) -> Tensor:
        if img.ndim != 4:
            raise ValueError("Input img tensor must be in [B, C, H, W] format.")
        if txt.ndim != 3:
            raise ValueError("Input txt tensors must have 3 dimensions.")
        B, C, H, W = img.shape

        device = img.device
        dtype = img.dtype
        nb_double_block = len(self.double_blocks)
        nb_single_block = len(self.single_blocks)

        # Store the raw pixel values of each patch for the NeRF head later.
        # unfold creates patches: [B, C * P * P, NumPatches]
        nerf_pixels = nn.functional.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)

        # patchify ops
        img = self.img_in_patch(img) # -> [B, Hidden, H/P, W/P]
        num_patches = img.shape[2] * img.shape[3]
        # flatten into a sequence for the transformer.
        img = img.flatten(2).transpose(1, 2) # -> [B, NumPatches, Hidden]

        txt = self.txt_in(txt)

        distill_timestep = timestep_embedding(timestep, self.approximator_in_dim//4)
        distill_guidance = timestep_embedding(guidance, self.approximator_in_dim//4)
        timestep_guidance = (
            torch.cat([distill_timestep, distill_guidance], dim=1)
            .unsqueeze(1)
            .repeat(1, self.mod_index_length, 1)
        )

        modulation_index = timestep_embedding(self.mod_index.to(timestep.device), self.approximator_in_dim//2)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        # and we need to broadcast timestep and guidance along too
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(dtype)
        mod_vectors = self.distilled_guidance_layer(input_vec)

        ids = torch.cat((txt_ids, img_ids), dim=1)

        if self.use_dynamicPE:
            self.pe_embedder.set_timestep(timestep.item())
            pes = []
            for i in range(ids.shape[0]):
                pe = self.pe_embedder(ids[i])

                out = torch.stack([pe[0], -pe[1], pe[1], pe[0]], dim=-1).unsqueeze(0)
                out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
                pes.append(out.unsqueeze(1))
            pe = torch.cat(pes, dim=0)
        else:
            pe = self.pe_embedder(ids)

        scratchQ = torch.empty((img.shape[0], 24, img.shape[1]+txt.shape[1], 128), device=device, dtype=dtype)   # preallocated for combined q|k|v_img|txt
        scratchK = torch.empty((img.shape[0], 24, img.shape[1]+txt.shape[1], 128), device=device, dtype=dtype)
        scratchV = torch.empty((img.shape[0], 24, img.shape[1]+txt.shape[1], 128), device=device, dtype=dtype)
        idx_i = 3 * nb_single_block
        idx_t = 3 * nb_single_block + 6 * nb_double_block
        for _i, block in enumerate(self.double_blocks):
            img_mod1 = mod_vectors[:, idx_i+0:idx_i+3, :]
            img_mod2 = mod_vectors[:, idx_i+3:idx_i+6, :]
            idx_i += 6
            txt_mod1 = mod_vectors[:, idx_t+0:idx_t+3, :]
            txt_mod2 = mod_vectors[:, idx_t+3:idx_t+6, :]
            idx_t += 6
            img, txt = block(scratchQ, scratchK, scratchV, img=img, txt=txt, img_mod1=img_mod1, img_mod2=img_mod2, txt_mod1=txt_mod1, txt_mod2=txt_mod2, pe=pe)
        del scratchQ, scratchK, scratchV

        img = torch.cat((txt, img), 1)

        scratchA = torch.empty((img.shape[0], img.shape[1], 15360), device=device, dtype=dtype)
        idx = 0
        for _i, block in enumerate(self.single_blocks):
            img = block(scratchA, img, shift=mod_vectors[:, idx+0:idx+1, :], scale=mod_vectors[:, idx+1:idx+2, :], gate=mod_vectors[:, idx+2:idx+3, :], pe=pe)
            idx += 3
        del scratchA

        # aliasing, reshape for per-patch processing
        nerf_hidden = img[:, txt.shape[1] :, ...].reshape(B * num_patches, self.hidden_size)
        nerf_pixels = rearrange(nerf_pixels, "b (c p) n -> (b n) p c", b=B, c=C, n=num_patches, p=self.patch_size**2) #transpose, reshape, transpose

        # get DCT-encoded pixel embeddings [pixel-dct]
        img_dct = self.nerf_image_embedder(nerf_pixels)

        # pass through the dynamic MLP blocks (the NeRF)
        for _i, block in enumerate(self.nerf_blocks):
            img_dct = block(img_dct, nerf_hidden)

        # final projection to get the output pixel values
        # img_dct = self.nerf_final_layer(img_dct) # -> [B*NumPatches, P*P, C]
        img_dct = self.nerf_final_layer_conv.norm(img_dct)

        # Reassemble the patches into the final image.
        img_dct = rearrange(img_dct, "(b n) p c -> b (c p) n", b=B, n=num_patches, p=self.patch_size**2) #transpose, reshape, transpose

        img_dct = nn.functional.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ) # [B, Hidden, H, W]
        img_dct = self.nerf_final_layer_conv.conv(img_dct)

        return img_dct

    def forward(self, x, timestep, context, **kwargs):
        """
        Forge-compatible forward method that adapts to the expected interface.

        Args:
            x: Input image tensor [B, C, H, W]
            timestep: Timestep tensor [B] or scalar
            context: Text conditioning tensor [B, seq_len, dim]
            **kwargs: Additional arguments (ignored)

        Returns:
            Output tensor [B, C, H, W]
        """
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype

        # Convert timestep to correct format
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).repeat(bs)
        elif timestep.shape[0] == 1 and bs > 1:
            timestep = timestep.repeat(bs)

        # Create image position IDs (similar to regular Chroma)
        patch_size = self.patch_size  # Use the actual patch size from params
        h_patches = h // patch_size
        w_patches = w // patch_size

        img_ids = torch.zeros((h_patches, w_patches, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_patches - 1, steps=h_patches, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_patches - 1, steps=w_patches, device=input_device, dtype=input_dtype)[None, :]
        img_ids = img_ids[None, :].repeat(bs, 1, 1, 1)
        img_ids = img_ids.reshape(bs, h_patches * w_patches, 3)

        # Create text position IDs (all zeros)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)

        # Create guidance (set to 0 for now, might need adjustment)
        guidance = torch.zeros((bs,), device=input_device, dtype=input_dtype)

        # Call the inner forward method
        result = self.inner_forward(
            img=x,
            img_ids=img_ids,
            txt=context,
            txt_ids=txt_ids,
            timestep=timestep,
            guidance=guidance,
        )

        return result

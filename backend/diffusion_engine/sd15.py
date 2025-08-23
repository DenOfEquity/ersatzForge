import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.args import dynamic_args
from backend import memory_management

import safetensors.torch as sf
from backend import utils

from modules import shared
from spaces import download_single_file

## ELLA
from collections import OrderedDict
from typing import Optional
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

class AdaLayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", torch.nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class PerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = torch.nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = torch.nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = torch.nn.Linear(input_dim, width)

        self.perceiver_blocks = torch.nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = torch.nn.Sequential(
                torch.nn.Linear(width, output_dim), torch.nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class ELLA(torch.nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        input_dim=2048,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states
## end: ELLA


class StableDiffusion(ForgeDiffusionEngine):
    matched_guesses = [model_list.SD15]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={
                'clip_l': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                'clip_l': huggingface_components['tokenizer']
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        unet = UnetPatcher.from_model(
            model=huggingface_components['unet'],
            diffusers_scheduler=huggingface_components['scheduler'],
            config=estimated_config
        )

        self.text_processing_engine = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_l',
            embedding_expected_shape=768,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # WebUI Legacy
        self.is_sd1 = True

        self.ella = None

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str], end_steps):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if "ELLA" in shared.opts.use_ELLA:
            if self.ella is None:
                self.ella = ELLA()

                ella_model_file = download_single_file(
                    "https://huggingface.co/QQGYLab/ELLA/resolve/main/ella-sd1.5-tsc-t5xl.safetensors", 
                    model_dir = "models\\ELLA",
                    progress = True,
                    file_name = "ella-sd1.5-tsc-t5xl.safetensors",
                )
                sf.load_model(self.ella, ella_model_file, strict=True)

            #use diffusers for t5xl, auto-download and nothing else uses it
            from transformers import T5EncoderModel, T5Tokenizer
            T5model = T5EncoderModel.from_pretrained("QQGYLab/ELLA", subfolder="models--google--flan-t5-xl--text_encoder")
            T5tokenizer = T5Tokenizer.from_pretrained("QQGYLab/ELLA", subfolder="models--google--flan-t5-xl--text_encoder", legacy=True)

#use memory_management.text_encoder_device ? and text_encoder_offload_device
#save T5, move to cpu after use ? would be relying on garbage collection after model change

            cond_ella = []
            timestep = 999
            start_step = 0

            for p in range(len(prompt)):
                text_inputs = T5tokenizer(prompt[p], return_tensors="pt", add_special_tokens=True)
                outputs = T5model(text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
                cond_t5 = outputs.last_hidden_state

                # prompt scheduling
                timestep = 999 * (1.0 - (start_step / end_steps[-1]) ** 2) # squared timesteps schedule
                # timestepL = 999 - (999 * start_step / end_steps[-1]) # linear timesteps schedule
                # timestepS = 999 * (1.0 - (start_step / end_steps[-1]) ** 0.5) # sqrt timesteps schedule

                cond_ella.append(self.ella(cond_t5, torch.Tensor([timestep]))) # seems best at 999 for normal use, but with prompt scheduling?

                start_step = end_steps[p]

            del T5model, T5tokenizer, cond_t5, text_inputs, outputs

            if "CLIP" in shared.opts.use_ELLA:
                cond_clip = self.text_processing_engine(prompt)
                cond = torch.cat([cond_clip, torch.cat(cond_ella, dim=0).to(cond_clip)], dim=1)
            else:
                cond = cond_ella
        else:
            cond = self.text_processing_engine(prompt)

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine.process_texts([prompt])
        return token_count, self.text_processing_engine.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    def save_checkpoint(self, filename):
        sd = {}
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model, prefix='model.diffusion_model.')
        )
        sd.update(
            model_list.SD15.process_clip_state_dict_for_saving(self, 
                utils.get_state_dict_after_quant(self.forge_objects.clip.cond_stage_model, prefix='')
            )
        )
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.vae.first_stage_model, prefix='first_stage_model.')
        )
        sf.save_file(sd, filename)
        return filename

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.gemma_engine import GemmaTextProcessingEngine

# from modules.shared import opts


class Lumina2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Lumina2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(model_dict={"gemma2": huggingface_components["text_encoder"]}, tokenizer_dict={"gemma2": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = PredictionDiscreteFlow(shift=6.0)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_gemma = GemmaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.gemma2,
            tokenizer=clip.tokenizer.gemma2,
        )

        self.is_lumina2 = True

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    # def set_clip_skip(self, clip_skip):
        # def sigma (timestep, s):
            # return s * timestep / (1 + (s - 1) * timestep)

        # ts = sigma((torch.arange(1, 10000 + 1, 1) / 10000), opts.lumina2_flow_shift)
        # self.forge_objects.unet.model.predictor.sigmas = ts

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond_gemma = self.text_processing_engine_gemma(prompt)
        cond = dict(crossattn=cond_gemma)
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_gemma.tokenize([prompt])[0])
        return token_count, max(256, token_count)

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

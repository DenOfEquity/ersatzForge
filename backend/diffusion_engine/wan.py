import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.t5_engine import T5TextProcessingEngine


class Wan(ForgeDiffusionEngine):
    matched_guesses = [model_list.WAN22_T2V, model_list.WAN21_T2V]#, model_list.WAN21_I2V]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict     = {"umt5xxl": huggingface_components["text_encoder"]},
            tokenizer_dict = {"umt5xxl": huggingface_components["tokenizer"]}
        )

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = PredictionDiscreteFlow(shift=5.0)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_t5 = T5TextProcessingEngine(
            text_encoder=clip.cond_stage_model.umt5xxl,
            tokenizer=clip.tokenizer.umt5xxl,
            min_length=512,
            add_special_tokens=True,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

    def set_clip_skip(self, clip_skip):
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = self.text_processing_engine_t5.tokenize_for_UI(prompt)
        return token_count, max(512, token_count)

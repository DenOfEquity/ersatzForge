import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import Qwen3TextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux2
from backend import memory_management


class ERNIE(ForgeDiffusionEngine):
    matched_guesses = [model_list.ERNIEImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={
                'mistral3': huggingface_components['text_encoder'],
            },
            tokenizer_dict={
                'mistral3': huggingface_components['tokenizer'],
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        k_predictor = PredictionFlux2(estimated_config)

        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        self.text_processing_engine = Qwen3TextProcessingEngine(
            text_encoder=clip.cond_stage_model.mistral3,
            tokenizer=clip.tokenizer.mistral3,
            is_ernie=True,
        )

        self.is_ernie = True
        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond_qwen, _ = self.text_processing_engine(prompt)
        return cond_qwen

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = self.text_processing_engine.tokenize_for_UI(prompt)
        return token_count, -1

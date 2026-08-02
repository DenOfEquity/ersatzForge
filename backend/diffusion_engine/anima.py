import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.anima_engine import AnimaTextProcessingEngine
from backend.modules.k_prediction import PredictionDiscreteFlow
# from backend.modules.k_prediction import PredictionCosmosRFlow
from backend import memory_management


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict = {
                "qwen3": huggingface_components["text_encoder"]
            },
            tokenizer_dict = {
                "qwen3": huggingface_components["tokenizer"], "t5xxl": huggingface_components["tokenizer_2"]
            }
        )

        vae = VAE(model=huggingface_components["vae"])
        k_predictor = PredictionDiscreteFlow(shift=3.0, multiplier=1.0)
        # k_predictor = PredictionCosmosRFlow(sigma_max=80.0)
        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        self.text_processing_engine_anima = AnimaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3,
            qwen_tokenizer=clip.tokenizer.qwen3,
            t5_tokenizer=clip.tokenizer.t5xxl,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_cosmos_predict2 = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond_anima, negpip = self.text_processing_engine_anima(prompt)
        if negpip is None:
            return cond_anima
        else:
            cond = dict(negpip=negpip, crossattn=cond_anima)
            return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = self.text_processing_engine_anima.tokenize_for_UI(prompt)
        return token_count, -1

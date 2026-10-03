import torch
import safetensors.torch as sf

from backend import utils
from modules.shared import opts
from modules_forge import colour_code as cc


class ForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeObjects(
            self.unet,
            self.clip,
            self.vae,
            self.clipvision
        )


class ForgeDiffusionEngine:
    matched_guesses = []

    def __init__(self, estimated_config, huggingface_components):
        self.model_config = estimated_config
        self.is_inpaint = estimated_config.inpaint_model()

        self.forge_objects = None
        self.forge_objects_original = None
        self.forge_objects_after_applying_lora = None

        self.current_lora_hash = str([])

        self.fix_for_webui_backward_compatibility()

    def set_shift(self, sequence_length):
        pass

    def apply_shift(self, option, sequence_length, max_sequence_length=4096, terminal=0.0):
        # called by set_shift() in actual diffusion engine
        if not hasattr(self, "original_sigmas"):
            self.original_sigmas = self.forge_objects.unet.model.predictor.sigmas.clone()
            self.sigmas_length = len(self.original_sigmas) # some Predictors use 1000, others 10000
            self.last_shift = (0, 0, 0)

        timesteps = None

        shift_parts = getattr(opts, option, "").strip()
        if shift_parts == "":
            timesteps = self.original_sigmas.clone()
            self.last_shift = (0, 0, 0)
        else:
            try:
                shift_parts = [float(s.strip()) for s in shift_parts.split(",")[0:2]]
                if len(shift_parts) == 1:
                    shift = max(0.25, shift_parts[0])
                    if self.last_shift[0] == shift:
                        return
                    base_shift = 0.0
                    max_shift = 0.0
                else:
                    base_shift = max(0.2, shift_parts[0])
                    max_shift = max(base_shift, shift_parts[1])
                    if self.last_shift[1] == base_shift and self.last_shift[2] == max_shift:
                        return
                    shift = 0.0
            except Exception:
                print (f"{cc.WARNING}[Shift]{cc.MINOR} Error parsing Setting{cc.RESET} '{option}' - using original sigmas.")
                timesteps = self.original_sigmas.clone()
                self.last_shift = (0, 0, 0)

        if timesteps is None:
            timesteps = torch.arange(1, self.sigmas_length + 1, 1) / self.sigmas_length

            if shift > 0.0:
                timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
            else:
                base_sequence_len = 256

                m = (max_shift - base_shift) / (max_sequence_length - base_sequence_len)
                b = base_shift - m * base_sequence_len
                mu = sequence_length * m + b
                mu = torch.tensor(mu).to(timesteps)
                timesteps = torch.exp(mu) / (torch.exp(mu) + (1 / timesteps - 1))

            if terminal > 0.0:
                one_minus_z = 1 - timesteps
                scale_factor = one_minus_z[0] / (1 - terminal)
                timesteps = 1 - (one_minus_z / scale_factor)

            self.last_shift = (shift, base_shift, max_shift)

        self.forge_objects.unet.model.predictor.register_buffer("sigmas", timesteps)

    def set_clip_skip(self, clip_skip):
        pass

    def get_first_stage_encoding(self, x):
        return x  # legacy code, do not change

    def get_learned_conditioning(self, prompt: list[str]):
        pass

    def get_prompt_lengths_on_ui(self, prompt):
        return 0, 75

    def is_webui_legacy_model(self):
        return self.is_sd1 or self.is_sd2 or self.is_sdxl or self.is_sd3

    def fix_for_webui_backward_compatibility(self):
        self.first_stage_model = None
        self.cond_stage_model = None
        self.use_distilled_cfg_scale = False
        self.is_sd1 = False
        self.is_sd2 = False
        self.is_sdxl = False
        self.is_sd3 = False
        self.is_cosmos_predict2 = False
        self.is_wan = False
        self.is_flux = False # also Chroma
        self.is_flux2 = False
        self.is_chromaDCT = False
        self.is_lumina2 = False
        self.is_ernie = False
        self.is_krea2 = False
        self.is_qwen21 = False

        return

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

    def save_unet(self, filename):
        sd = utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model)
        sf.save_file(sd, filename)
        return filename

    def save_checkpoint(self, filename):
        sd = {}
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model, prefix='model.diffusion_model.')
        )
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.clip.cond_stage_model, prefix='text_encoders.')
        )
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.vae.first_stage_model, prefix='vae.')
        )
        sf.save_file(sd, filename)
        return filename

from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_controllllite.lib_controllllite import LLLiteLoader


opLLLiteLoader = LLLiteLoader().load_lllite


class ControlLLLitePatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any('lllite' in k for k in state_dict.keys()):
            return None
        return ControlLLLitePatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        if mask is not None:
            cond *= mask

        num_steps = process.steps
        start_step = int(num_steps * self.start_percent)
        end_step = int(num_steps * self.end_percent)

        unet = opLLLiteLoader(
            model=unet,
            state_dict=self.state_dict,
            cond_image=cond,
            strength=self.strength,
            start_step=start_step,
            end_step=end_step,
        )

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(ControlLLLitePatcher)

import gradio as gr

from modules import scripts
from backend import memory_management


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled = 'Normal'
        self.original_vram_state = memory_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Radio(label='For UNet', choices=['Normal', 'Offload nn.Linear modules', 'Maximize offload'], value='Normal')
            vae_enabled = gr.Checkbox(label='Enabled for VAE (always tiled)', value=False)
            with gr.Row():
                tile_size_x  = gr.Slider(label="Tile width (pixels)",   value=512, minimum=128, maximum=1536, step=16)
                tile_size_y  = gr.Slider(label="Tile height (pixels)",  value=512, minimum=128, maximum=1536, step=16)
            with gr.Row():
                tile_overlap = gr.Slider(label="Tile overlap (pixels)", value=64,  minimum=16,  maximum=256,  step=16)
                tile_method = gr.Radio(label="Tile decode method", choices=["original", "diffusers", "DoE"], value="original")

        return unet_enabled, vae_enabled, tile_size_x, tile_size_y, tile_overlap, tile_method

    def process(self, p, *script_args, **kwargs):
        unet_enabled, vae_enabled, tile_size_x, tile_size_y, tile_overlap, tile_method = script_args

        if unet_enabled != 'Normal':
            print(f'NeverOOM Enabled for UNet ({unet_enabled})')

        if vae_enabled:
            print('NeverOOM Enabled for VAE (always tiled)')
            p.sd_model.forge_objects.vae.tile_info = (tile_size_x, tile_size_y, tile_overlap, tile_method)
        else:
            p.sd_model.forge_objects.vae.tile_info = None

        memory_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            memory_management.unload_all_models()
            match unet_enabled:
                case 'Offload nn.Linear modules':
                    self.original_vram_state = memory_management.vram_state
                    memory_management.vram_state = memory_management.VRAMState.VERY_LOW_VRAM
                case 'Maximize offload':
                    self.original_vram_state = memory_management.vram_state
                    memory_management.vram_state = memory_management.VRAMState.NO_VRAM
                case _:
                    memory_management.vram_state = self.original_vram_state

            print(f'VRAM State Changed To {memory_management.vram_state.name}')
            self.previous_unet_enabled = unet_enabled

        return

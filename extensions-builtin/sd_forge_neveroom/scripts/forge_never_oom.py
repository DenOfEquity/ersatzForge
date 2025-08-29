import gradio as gr

from modules import scripts
from backend import memory_management


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled = False
        self.original_vram_state = memory_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Checkbox(label='Enabled for UNet (always maximize offload)', value=False)
            vae_enabled = gr.Checkbox(label='Enabled for VAE (always tiled)', value=False)
            with gr.Row():
                tile_size_x  = gr.Slider(label="Tile width (pixels)",   value=512, minimum=128, maximum=1536, step=64)
                tile_size_y  = gr.Slider(label="Tile height (pixels)",  value=512, minimum=128, maximum=1536, step=64)
                tile_overlap = gr.Slider(label="Tile overlap (pixels)", value=64,  minimum=16,  maximum=256,  step=16)
        return unet_enabled, vae_enabled, tile_size_x, tile_size_y, tile_overlap

    def process(self, p, *script_args, **kwargs):
        unet_enabled, vae_enabled, tile_size_x, tile_size_y, tile_overlap = script_args

        if unet_enabled:
            print('NeverOOM Enabled for UNet (always maximize offload)')

        if vae_enabled:
            print('NeverOOM Enabled for VAE (always tiled)')
            p.sd_model.forge_objects.vae.tile_size = (tile_size_x, tile_size_y, tile_overlap)
        else:
            p.sd_model.forge_objects.vae.tile_size = None

        memory_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            memory_management.unload_all_models()
            if unet_enabled:
                self.original_vram_state = memory_management.vram_state
                memory_management.vram_state = memory_management.VRAMState.NO_VRAM
            else:
                memory_management.vram_state = self.original_vram_state
            print(f'VRAM State Changed To {memory_management.vram_state.name}')
            self.previous_unet_enabled = unet_enabled

        return

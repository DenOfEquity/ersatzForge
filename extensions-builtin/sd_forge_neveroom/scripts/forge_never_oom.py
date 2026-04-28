import gradio as gr

from modules import scripts
from backend import memory_management

import modules_forge.colour_code as cc


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
            unet_enabled = gr.Radio(label="For UNet", choices=["Normal", "Offload nn.Linear modules", "Maximize offload"], value='Normal')
            vae_enabled = gr.Checkbox(label="Enabled for VAE (always tiled)", value=False)
            with gr.Row():
                tile_size_x  = gr.Slider(label="Tile width (pixels)",   value=512, minimum=128, maximum=1536, step=16)
                tile_size_y  = gr.Slider(label="Tile height (pixels)",  value=512, minimum=128, maximum=1536, step=16)
            with gr.Row():
                tile_overlap = gr.Slider(label="Tile overlap (pixels)", value=64,  minimum=16,  maximum=256,  step=16)
                tile_method = gr.Radio(label="Tile decode method", choices=["original", "diffusers", "DoE"], value="original")
            controlnet_on_cpu = gr.Checkbox(label="Enabled for ControlNet (stored on CPU)", value=False)
            with gr.Row():
                tiled_conv2d = gr.Dropdown(label="Tiled conv2d", choices=["Disabled", "64", "96", "128"], value="Disabled", type="value", scale=0)
        return unet_enabled, vae_enabled, controlnet_on_cpu, tile_size_x, tile_size_y, tile_overlap, tile_method, tiled_conv2d

    def process(self, p, *script_args, **kwargs):
        unet_enabled, vae_enabled, controlnet_on_cpu, tile_size_x, tile_size_y, tile_overlap, tile_method, tiled_conv2d = script_args

        if unet_enabled != 'Normal':
            print(f"{cc.SETTING}NeverOOM Enabled for UNet ({unet_enabled}){cc.RESET}")

        if vae_enabled:
            print(f"{cc.SETTING}NeverOOM Enabled for VAE (always tiled){cc.RESET}")
            p.sd_model.forge_objects.vae.tile_info = (tile_size_x, tile_size_y, tile_overlap, tile_method)
            p.extra_generation_params.update({
                "tiled VAE"        : (tile_size_x, tile_size_y, tile_overlap, tile_method),
            })  # informational only
        else:
            p.sd_model.forge_objects.vae.tile_info = None

        if controlnet_on_cpu:
            print(f"{cc.SETTING}NeverOOM Enabled for ControlNets (always on CPU){cc.RESET}")
        memory_management.controlnet_on_cpu = controlnet_on_cpu

        memory_management.VAE_ALWAYS_TILED = vae_enabled

        match tiled_conv2d:
            case "64":
                memory_management.tiled_conv2d = 64
            case "96":
                memory_management.tiled_conv2d = 96
            case "128":
                memory_management.tiled_conv2d = 128
            case _:
                memory_management.tiled_conv2d = 0
        if memory_management.tiled_conv2d != 0:
            print(f"{cc.SETTING}NeverOOM Enabled tiled conv2d{cc.RESET}")

        if self.previous_unet_enabled != unet_enabled:
            memory_management.unload_all_models()
            match unet_enabled:
                case "Offload nn.Linear modules":
                    memory_management.vram_state = memory_management.VRAMState.VERY_LOW_VRAM
                case "Maximize offload":
                    memory_management.vram_state = memory_management.VRAMState.NO_VRAM
                case _:
                    memory_management.vram_state = self.original_vram_state

            print(f"{cc.SETTING}Set VRAM state to: {memory_management.vram_state.name}{cc.RESET}")
            self.previous_unet_enabled = unet_enabled

        return

import gradio
import torch
import numpy
from einops import rearrange

from modules import scripts, shared
from modules.api.api import decode_base64_to_image
from modules.ui_components import InputAccordion, ToolButton
from modules.sd_samplers_common import images_tensor_to_samples
from backend.misc.image_resize import adaptive_resize


class ersatzZImageTurboControl(scripts.Script):
    sorting_priority = 0
    last_image_hash = None
    last_latent_size = None

    def title(self):
        return "Z-Image-Turbo Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            gradio.Markdown("Select the control model in the **Additional modules** menu. Include pre-processed reference image here.")
            with gradio.Row():
                with gradio.Column():
                    zitc_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                with gradio.Column():
                    image_str = gradio.Slider(value=1.0, minimum=0.0, maximum=2.0, step=0.01, label="strength")
                    image_stop = gradio.Slider(value=0.75, minimum=0.0, maximum=1.0, step=0.01, label="stop sigma")
                    with gradio.Row():
                        image_info = gradio.Textbox(value="", show_label=False, interactive=False, max_lines=1)
                        image_send = ToolButton(value="\U0001F4D0", interactive=False, variant="tertiary")
                        image_dims = gradio.Textbox(visible=False, value="0")

                def get_dims(image):
                    if image:
                        w = image.size[0]
                        h = image.size[1]
                        sw = 16 * ((15 + w) // 16)
                        sh = 16 * ((15 + h) // 16)
                        return f"{image.size[0]} × {image.size[1]} ({sw} × {sh})", gradio.update(interactive=True, variant="secondary"), f"{sw},{sh}"
                    else:
                        return  "", gradio.update(interactive=False, variant='tertiary'), "0"

                zitc_image1.change(fn=get_dims, inputs=zitc_image1, outputs=[image_info, image_send, image_dims], show_progress=False)
 
                if self.is_img2img:
                    tab_id = gradio.State(value="img2img")
                else:
                    tab_id = gradio.State(value="txt2img")
 
                image_send.click(fn=None, js="kontext_set_dimensions", inputs=[tab_id, image_dims], outputs=None)

        self.infotext_fields = [
            (enabled,  lambda d: d.get("zitc_enabled", False)),
            (image_str,  "zitc_strength"),
            (image_stop, "zitc_stop"),
        ]

        return enabled, zitc_image1, image_str, image_stop


    def process(self, params, *script_args, **kwargs):
        enabled, image, strength, stop = script_args
        if enabled and image is not None and strength > 0.0 and stop > 0.0 and params.sd_model.is_lumina2:
            if getattr(shared.sd_model.forge_objects.unet.model.diffusion_model, "control", False):
                params.extra_generation_params.update(dict(
                    zitc_enabled  = enabled,
                    zitc_strength = strength,
                    zitc_stop     = stop,
                ))
            else:
                print ("[Z-Image-Turbo Control] Control model not loaded.")

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, image, strength, stop = script_args

        if enabled and image is not None and strength > 0.0 and stop > 0.0 and params.sd_model.is_lumina2 and getattr(shared.sd_model.forge_objects.unet.model.diffusion_model, "control", False):
            if params.iteration > 0:    # batch count
                # setup done on iteration 0
                return

            x = kwargs["x"]
            _, c, h, w = x.size()
            
            def calc_extra_mem(latent):
                return latent.shape[0] * latent.shape[1] * latent.shape[2] * latent.element_size() * 1024

            this_image_hash = hash(str(list(image.getdata(band=None))).encode("utf-8"))
            this_latent_size = (w, h)

            if this_image_hash == self.last_image_hash and this_latent_size == self.last_latent_size:
                print ("[Z-Image-Turbo Control] used cache")
                shared.ZITstrength = strength
                shared.ZITstop = stop
                extra_mem = calc_extra_mem(shared.ZITlatent)
            else:
                self.last_image_hash = this_image_hash
                self.last_latent_size = this_latent_size

                input_device = x.device
                input_dtype = x.dtype

                if isinstance (image, str):
                    k_image = decode_base64_to_image(image).convert("RGB")
                else:
                    k_image = image.convert("RGB")

                k_image = numpy.array(k_image) / 255.0
                k_image = numpy.transpose(k_image, (2, 0, 1))
                k_image = torch.tensor(k_image).unsqueeze(0)

                # only go through the resize process if image is not already desired size
                k_width = w * 8
                k_height = h * 8

                if k_image.shape[3] != k_width or k_image.shape[2] != k_height:
                    print ("[Z-Image-Turbo Control] resizing and center-cropping input to: ", k_width, k_height)
                    k_image = adaptive_resize(k_image, k_width, k_height, "lanczos", "center")
                else:
                    print ("[Z-Image-Turbo Control] no image resize needed")

                k_latent = images_tensor_to_samples(k_image, None, None)

                # pad if needed - latent width and height must be multiple of 2
                # could just adjust the resize to be *16, but the padding might be better for images that need only one extra row/col
                patch_size = 2
                pad_h = k_latent.shape[2] % patch_size
                pad_w = k_latent.shape[3] % patch_size
                k_latent = torch.nn.functional.pad(k_latent, (0, pad_w, 0, pad_h), mode="circular")

                k_latent = rearrange(k_latent, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=2, pw=2)
                k_latent = k_latent.to(input_device, input_dtype)

                shared.ZITlatent = k_latent
                shared.ZITstrength = strength
                shared.ZITstop = stop

                extra_mem = calc_extra_mem(k_latent)

            print ("[Z-Image-Turbo Control] reserving extra memory (MB):", round(extra_mem/(1024*1024), 2))
            params.sd_model.forge_objects.unet.extra_preserved_memory_during_sampling = extra_mem

        return

    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            # shared.ZITlatent = None # if don't clear, can avoid VAE encode if image not changed
            shared.ZITstrength = 0.0
            shared.ZITstop = 0.0
            params.sd_model.forge_objects.unet.extra_preserved_memory_during_sampling = 0

        return

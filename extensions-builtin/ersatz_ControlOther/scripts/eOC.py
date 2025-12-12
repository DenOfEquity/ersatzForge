import gradio
import torch
import numpy
from einops import rearrange, repeat

from modules import scripts, shared
from modules.api.api import decode_base64_to_image
from modules.ui_components import InputAccordion, ToolButton
from modules.sd_samplers_common import images_tensor_to_samples
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import IntegratedFluxTransformer2DModel


def patched_flux_forward(self, x, timestep, context, y, guidance=None, **kwargs):
    bs, c, h, w = x.shape

    if c != 16:
        # fix the case where user is also using FluxTools extension, x has extra channels
        # spam message every step, so user might pay attention, or silently fix?
        # print ("\n[Kontext] ERROR: too many channels, excess channels will be stripped.\n")
        x = x[:, :16, :, :]

    input_device = x.device
    input_dtype = x.dtype
    patch_size = 2
    pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
    pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    del x, pad_h, pad_w
    h_len = ((h + (patch_size // 2)) // patch_size)
    w_len = ((w + (patch_size // 2)) // patch_size)

    img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    img_tokens = img.shape[1]

    if ersatzOtherControl.kontext_latent is not None:
        img = torch.cat([img, ersatzOtherControl.kontext_latent.repeat(bs, 1, 1)], dim=1)
        img_ids = torch.cat([img_ids, ersatzOtherControl.kontext_ids.repeat(bs, 1, 1)], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
    del input_device, input_dtype
    out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance)
    del img, img_ids, txt_ids, timestep, context

    out = out[:, :img_tokens]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]

    del h_len, w_len, bs

    return out


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),    (688, 1504),    (720, 1456),    (752, 1392),    (800, 1328),    (832, 1248),    (880, 1184),    (944, 1104),
    (1024, 1024),
    (1104, 944),    (1184, 880),    (1248, 832),    (1328, 800),    (1392, 752),    (1456, 720),    (1504, 688),    (1568, 672),
]


#extra memory reservation for Z-Image-Turbo Controlommented out as control layers are processed before standard layers
#Task Manager shows increased VRAM usage, into shared memory, but no slowdown noticed


class ersatzOtherControl(scripts.Script):
    sorting_priority = 0
    zitc_image_hash = None
    zitc_latent_size = None
    original_kontext_forward = None
    kontext_latent = None
    kontext_ids = None
    kontext_image_hash = None
    kontext_latent_size = None
    kontext_resize = None

    def __init__(self):
        if ersatzOtherControl.original_kontext_forward is None:
            ersatzOtherControl.original_kontext_forward = IntegratedFluxTransformer2DModel.forward

    def title(self):
        return "(Other) Control Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        selected_tab = gradio.State(value=0)
        if self.is_img2img:
            tab_id = gradio.State(value="img2img")
        else:
            tab_id = gradio.State(value="txt2img")

        def get_dims(image):
            if image:
                w = image.size[0]
                h = image.size[1]
                sw = 16 * ((15 + w) // 16)
                sh = 16 * ((15 + h) // 16)
                return f"{image.size[0]} × {image.size[1]} ({sw} × {sh})", gradio.update(interactive=True, variant="secondary"), f"{sw},{sh}"
            else:
                return  "", gradio.update(interactive=False, variant='tertiary'), "0"

        with InputAccordion(False, label=self.title()) as enabled:
            with gradio.Tabs():
                with gradio.Tab("FluxKontext") as fkon:
                    gradio.Markdown("Select a FluxKontext model in the **Checkpoint** menu. Add reference image(s) here.")
                    with gradio.Row():
                        with gradio.Column():
                            k_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                            with gradio.Row():
                                k_image1_info = gradio.Textbox(value="", show_label=False, interactive=False, max_lines=1)
                                k_image1_send = ToolButton(value='\U0001F4D0', interactive=False, variant='tertiary')
                                k_image1_dims = gradio.Textbox(visible=False, value='0')
                        with gradio.Column():
                            k_image2 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                            with gradio.Row():
                                swap12 = ToolButton("\U000021C4")
                                k_image2_info = gradio.Textbox(value="", show_label=False, interactive=False, max_lines=1)
                                k_image2_send = ToolButton(value='\U0001F4D0', interactive=False, variant='tertiary')
                                k_image2_dims = gradio.Textbox(visible=False, value='0')

                        k_image1.change(fn=get_dims, inputs=k_image1, outputs=[k_image1_info, k_image1_send, k_image1_dims], show_progress="hidden")
                        k_image2.change(fn=get_dims, inputs=k_image2, outputs=[k_image2_info, k_image2_send, k_image2_dims], show_progress="hidden")
                        k_image1_send.click(fn=None, js="eOC_set_dimensions", inputs=[tab_id, k_image1_dims], outputs=None)
                        k_image2_send.click(fn=None, js="eOC_set_dimensions", inputs=[tab_id, k_image2_dims], outputs=None)

                    with gradio.Row():
                        kontext_sizing = gradio.Dropdown(label="Kontext image size/crop", choices=["no change", "to output", "to BFL recommended"], value="to BFL recommended")
                        kontext_reduce = gradio.Checkbox(False, info="This reduction is independent of the size/crop setting.", label="reduce to half width and height")

                    def kontext_swap(imageA, imageB):
                        return imageB, imageA
                    swap12.click(fn=kontext_swap, inputs=[k_image1, k_image2], outputs=[k_image1, k_image2])

                with gradio.Tab("Z-Image-Turbo Control") as zitc:
                    gradio.Markdown("Select the control model in the **Additional modules** menu. Include pre-processed reference image here.")
                    with gradio.Row():
                        with gradio.Column():
                            z_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                        with gradio.Column():
                            zitc_str = gradio.Slider(value=1.0, minimum=0.0, maximum=2.0, step=0.01, label="strength")
                            zitc_stop = gradio.Slider(value=0.75, minimum=0.0, maximum=1.0, step=0.01, label="stop sigma")
                            with gradio.Row():
                                z_image1_info = gradio.Textbox(value="", show_label=False, interactive=False, max_lines=1)
                                z_image1_send = ToolButton(value="\U0001F4D0", interactive=False, variant="tertiary")
                                z_image1_dims = gradio.Textbox(visible=False, value="0")

                        z_image1.change(fn=get_dims, inputs=z_image1, outputs=[z_image1_info, z_image1_send, z_image1_dims], show_progress="hidden")
                        z_image1_send.click(fn=None, js="eOC_set_dimensions", inputs=[tab_id, z_image1_dims], outputs=None)


            fkon.select(fn=lambda: 0, inputs=None, outputs=selected_tab, show_progress="hidden")
            zitc.select(fn=lambda: 1, inputs=None, outputs=selected_tab, show_progress="hidden")


        self.infotext_fields = [
            (enabled,  lambda d: d.get("eOC_enabled", False)),
            (zitc_str,  "zitc_strength"),
            (zitc_stop, "zitc_stop"),
            (kontext_sizing, "kontext_sizing"),
            (kontext_reduce, "kontext_reduce"),
        ]

        return enabled, selected_tab, z_image1, zitc_str, zitc_stop, k_image1, k_image2, kontext_sizing, kontext_reduce


    def process(self, params, *script_args, **kwargs):
        enabled, selected_tab, zitc_image, zitc_strength, zitc_stop, kontext_image1, kontext_image2, kontext_sizing, kontext_reduce = script_args
        if enabled:
            if selected_tab == 0 and (kontext_image1 is not None or kontext_image2 is not None) and params.sd_model.is_flux:
                params.extra_generation_params.update(dict(
                    eOC_enabled  = enabled,
                    kontext_sizing = kontext_sizing,
                    kontext_reduce = kontext_reduce,
                ))
            if selected_tab == 1 and zitc_image is not None and zitc_strength > 0.0 and zitc_stop < 1.0 and params.sd_model.is_lumina2:
                if getattr(shared.sd_model.forge_objects.unet.model.diffusion_model, "control", False):
                    params.extra_generation_params.update(dict(
                        eOC_enabled  = enabled,
                        zitc_strength = zitc_strength,
                        zitc_stop     = zitc_stop,
                    ))
                else:
                    print ("[Z-Image-Turbo Control] Control model not loaded.")


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, selected_tab, zitc_image, zitc_strength, zitc_stop, kontext_image1, kontext_image2, kontext_sizing, kontext_reduce = script_args

        if not enabled:
            return
        if params.iteration > 0:    # batch count
            # setup done on iteration 0
            return

        x = kwargs["x"]
        n, c, h, w = x.size()
        input_device = x.device
        input_dtype = x.dtype

        if selected_tab == 0 and (kontext_image1 is not None or kontext_image2 is not None) and params.sd_model.is_flux:
            imgs_data  = str(list(kontext_image1.getdata(band=None))) if kontext_image1 is not None else ""
            imgs_data += str(list(kontext_image2.getdata(band=None))) if kontext_image2 is not None else ""
            kontext_image_hash = hash(imgs_data)
            kontext_latent_size = (w, h)

            if kontext_image_hash == self.kontext_image_hash and kontext_latent_size == self.kontext_latent_size and self.kontext_resize == (kontext_sizing, kontext_reduce):
                print ("[Kontext] used cache")
            else:
                self.kontext_image_hash = kontext_image_hash
                self.kontext_latent_size = kontext_latent_size
                self.kontext_resize = (kontext_sizing, kontext_reduce)

                k_latents = []
                k_ids = []
                accum_h = 0
                accum_w = 0

                for image in [kontext_image1, kontext_image2]:
                    if image is not None:
                        if isinstance (image, str):
                            k_image = decode_base64_to_image(image).convert('RGB')
                        else:
                            k_image = image.convert('RGB')
                        k_image = numpy.array(k_image) / 255.0
                        k_image = numpy.transpose(k_image, (2, 0, 1))
                        k_image = torch.tensor(k_image).unsqueeze(0)

                        # it seems that the img_id is always 1 for the context images

                        # resize and combine here instead of in the forward function
                        # only go through the resize process if image is not already desired size
                        match kontext_sizing:
                            case "no change":
                                k_width = k_image.shape[3]
                                k_height = k_image.shape[2]
                            case "to output":
                                k_width = w * 8
                                k_height = h * 8
                            case "to BFL recommended":  # this snippet from ComfyUI
                                k_width = k_image.shape[3]
                                k_height = k_image.shape[2]
                                aspect_ratio = k_width / k_height
                                _, k_width, k_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)

                        if kontext_reduce:
                            k_width //= 2
                            k_height //= 2

                        if k_image.shape[3] != k_width or k_image.shape[2] != k_height:
                            print ("[Kontext] resizing and center-cropping input to: ", k_width, k_height)
                            k_image = adaptive_resize(k_image, k_width, k_height, "lanczos", "center")
                        else:
                            print ("[Kontext] no image resize needed")

                        # VAE encode each input image - combined image could be large
                        k_latent = images_tensor_to_samples(k_image, None, None)

                        # pad if needed - latent width and height must be multiple of 2
                        # could just adjust the resize to be *16, but the padding might be better for images that need only one extra row/col
                        patch_size = 2
                        pad_h = k_latent.shape[2] % patch_size
                        pad_w = k_latent.shape[3] % patch_size
                        k_latent = torch.nn.functional.pad(k_latent, (0, pad_w, 0, pad_h), mode="circular")

                        k_latents.append(rearrange(k_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size))
                        # imgs are combined in rearranged dimension 1 - so width/height can be independant of main latent and other inputs

                        latentH = k_latent.shape[2]
                        latentW = k_latent.shape[3]
                       
                        kh_len = ((latentH + (patch_size // 2)) // patch_size)
                        kw_len = ((latentW + (patch_size // 2)) // patch_size)

                        # this offset + accumulation is based on Comfy.
                        offset_h = 0
                        offset_w = 0
                        if kh_len + accum_h > kw_len + accum_w:
                            offset_w = accum_w
                        else:
                            offset_h = accum_h

                        k_id = torch.zeros((kh_len, kw_len, 3), device=input_device, dtype=input_dtype)
                        k_id[:, :, 0] = 1
                        k_id[:, :, 1] += torch.linspace(offset_h, offset_h + kh_len - 1, steps=kh_len, device=input_device, dtype=input_dtype)[:, None]
                        k_id[:, :, 2] += torch.linspace(offset_w, offset_w + kw_len - 1, steps=kw_len, device=input_device, dtype=input_dtype)[None, :]

                        accum_w = max(accum_w, kw_len + offset_w)
                        accum_h = max(accum_h, kh_len + offset_h)

                        k_ids.append(repeat(k_id, "h w c -> b (h w) c", b=1)) # moved batch into patched_flux_forward

                ersatzOtherControl.kontext_latent = torch.cat(k_latents, dim=1).contiguous().to(device=input_device, dtype=input_dtype)
                ersatzOtherControl.kontext_ids = torch.cat(k_ids, dim=1).to(device=input_device, dtype=input_dtype)

                del k_latent, k_id

            IntegratedFluxTransformer2DModel.forward = patched_flux_forward

            extra_mem = n * ersatzOtherControl.kontext_latent.shape[1] * ersatzOtherControl.kontext_latent.shape[2] * x.element_size() * 1024 #*1.65?
            print ("[Kontext] reserving extra memory (MB):", round(extra_mem/(1024*1024), 2))
            params.sd_model.forge_objects.unet.extra_preserved_memory_during_sampling = extra_mem


        if selected_tab == 1 and zitc_image is not None and zitc_strength > 0.0 and zitc_stop < 1.0 and params.sd_model.is_lumina2 and getattr(shared.sd_model.forge_objects.unet.model.diffusion_model, "control", False):

            # def calc_extra_mem(latent):
                # return latent.shape[0] * latent.shape[1] * latent.shape[2] * latent.element_size() * 1024

            zitc_image_hash = hash(str(list(zitc_image.getdata(band=None))))
            zitc_latent_size = (w, h)

            if zitc_image_hash == self.zitc_image_hash and zitc_latent_size == self.zitc_latent_size:
                print ("[Z-Image-Turbo Control] used cache")
                shared.ZITstrength = zitc_strength
                shared.ZITstop = zitc_stop
                # extra_mem = calc_extra_mem(shared.ZITlatent)
            else:
                self.zitc_image_hash = zitc_image_hash
                self.zitc_latent_size = zitc_latent_size

                if isinstance (zitc_image, str):
                    z_image = decode_base64_to_image(zitc_image).convert("RGB")
                else:
                    z_image = zitc_image.convert("RGB")

                z_image = numpy.array(z_image) / 255.0
                z_image = numpy.transpose(z_image, (2, 0, 1))
                z_image = torch.tensor(z_image).unsqueeze(0)

                # only go through the resize process if image is not already desired size
                z_width = w * 8
                z_height = h * 8

                if z_image.shape[3] != z_width or z_image.shape[2] != z_height:
                    print ("[Z-Image-Turbo Control] resizing and center-cropping input to: ", z_width, z_height)
                    z_image = adaptive_resize(z_image, z_width, z_height, "lanczos", "center")
                else:
                    print ("[Z-Image-Turbo Control] no image resize needed")

                z_latent = images_tensor_to_samples(z_image, None, None)

                # pad if needed - latent width and height must be multiple of 2
                patch_size = 2
                pad_h = z_latent.shape[2] % patch_size
                pad_w = z_latent.shape[3] % patch_size
                z_latent = torch.nn.functional.pad(z_latent, (0, pad_w, 0, pad_h), mode="circular")

                z_latent = rearrange(z_latent, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=2, pw=2)

                shared.ZITlatent = z_latent.contiguous().to(input_device, input_dtype)
                shared.ZITstrength = zitc_strength
                shared.ZITstop = zitc_stop

                # extra_mem = calc_extra_mem(k_latent)

            # print ("[Z-Image-Turbo Control] reserving extra memory (MB):", round(extra_mem/(1024*1024), 2))
            # params.sd_model.forge_objects.unet.extra_preserved_memory_during_sampling = extra_mem

        return

    def postprocess(self, params, processed, *args):
        enabled, selected_tab = args[0], args[1]
        if enabled:
            if selected_tab == 0:
                # self.kontext_latent = None
                # self.kontext_ids = None
                IntegratedFluxTransformer2DModel.forward = ersatzOtherControl.original_kontext_forward
                params.sd_model.forge_objects.unet.extra_preserved_memory_during_sampling = 0
            if selected_tab == 1:
                shared.ZITstrength = 0.0
                shared.ZITstop = 0.0
                # params.sd_model.forge_objects.unet.extra_preserved_memory_during_sampling = 0

        return


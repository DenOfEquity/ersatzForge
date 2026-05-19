import numpy
import torch
import gradio as gr
import os
from modules import scripts, shared
import torchvision.transforms.functional as TF

from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.ui_components import InputAccordion, ToolButton

try:
    import customPresets as colourPresets
except Exception:
    import colourPresets

class ColourNoiseForge(scripts.Script):
    centreNoise = False
    sharpNoise = False
    lowDNoise = False
    targetted = True
    everyStep = False
    latent = None

    from colourPresets import presetList

    def title(self):
        return "the Colour of Noise"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown("Settings are *very* dependent on model. Targetted noise generally tolerates higher strength and is easier to work with.")
            with gr.Row():
                delPreset = ToolButton(value="-", variant="secondary", tooltip="remove preset", elem_id="cn_b_remove")
                preset = gr.Dropdown(sorted([i[0] for i in self.presetList]), value="(None)", type="value", label="Colour presets", allow_custom_value=True)
                addPreset = ToolButton(value="+", variant="secondary", tooltip="add preset (no overwrite)", elem_id="cn_b_add")
                savePreset = ToolButton(value="save", variant="secondary", tooltip="save presets", elem_id="cn_b_save")

                ToolButton(value="️|", variant="tertiary", interactive=False)

                centreNoise = ToolButton(value="⨁", variant="secondary", tooltip="Centre initial noise", elem_id="cn_b_centre")
                lowDNoise = ToolButton(value="∿", variant="secondary", tooltip="Low Discrepancy noise", elem_id="cn_b_lowd")
                sharpNoise = ToolButton(value="♯", variant="secondary", tooltip="Sharpen initial noise", elem_id="cn_b_sharpen")
                targetNoise = ToolButton(value="🎯", variant="primary", tooltip="Targetted colour noise application", elem_id="cn_b_target")
                everyStep = ToolButton(value="E", variant="secondary", tooltip="Apply colour every step", elem_id="cn_b_every")

            with gr.Row():
                initialNoise = gr.ColorPicker(value="#000000", label="Colour", scale=0)
                noiseStrength = gr.Slider(minimum=0, maximum=0.2, value=0.0, step=0.001, label="strength")

                stepApplyP1 = gr.Slider(minimum=0, maximum=1.0, value=0, step=0.01, label="first pass step")
                stepApplyP2 = gr.Slider(minimum=0, maximum=1.0, value=0, step=0.01, label="high res step")

            def updateColours (preset):
                for i in range(len(self.presetList)):
                    p = self.presetList[i]
                    if p[0] == preset:
                        result = [p[1], p[2], p[3], p[4]]
                        ColourNoiseForge.centreNoise = not (p[5] & 1)
                        ColourNoiseForge.lowDNoise   = not (p[5] & 2)
                        ColourNoiseForge.sharpNoise  = not (p[5] & 4)
                        ColourNoiseForge.targetted   = not (p[5] & 8)
                        ColourNoiseForge.everyStep   = not (p[5] & 16)
                        result.append(toggleCentre())
                        result.append(togglelowD())
                        result.append(toggleSharp())
                        result.append(toggleTarget())
                        result.append(toggleEvery())
                        return result
                return [gr.skip()] * 9

            preset.change(fn=updateColours, inputs=[preset], outputs=[initialNoise, noiseStrength, stepApplyP1, stepApplyP2, centreNoise, lowDNoise, sharpNoise, targetNoise, everyStep], show_progress="hidden")

            # initialNoise.input(fn=lambda: "(custom)", inputs=None, outputs=[preset], show_progress="hidden")

            def toggleCentre ():
                ColourNoiseForge.centreNoise ^= True
                return gr.update(variant="primary" if ColourNoiseForge.centreNoise else "secondary")
            def togglelowD ():
                ColourNoiseForge.lowDNoise ^= True
                return gr.update(variant="primary" if ColourNoiseForge.lowDNoise else "secondary")
            def toggleSharp ():
                ColourNoiseForge.sharpNoise ^= True
                return gr.update(variant="primary" if ColourNoiseForge.sharpNoise else "secondary")
            def toggleTarget ():
                ColourNoiseForge.targetted ^= True
                return gr.update(variant="primary" if ColourNoiseForge.targetted else "secondary")
            def toggleEvery ():
                ColourNoiseForge.everyStep ^= True
                return gr.update(variant="primary" if ColourNoiseForge.everyStep else "secondary")

            def addColourPreset (name, c, s, s1, s2):
                if any(i[0] == name for i in self.presetList): # no direct overwrite - delete + add instead
                    return gr.skip()

                t = 0
                if ColourNoiseForge.centreNoise:
                    t += 1
                if ColourNoiseForge.lowDNoise:
                    t += 2
                if ColourNoiseForge.sharpNoise:
                    t += 4
                if ColourNoiseForge.targetted:
                    t += 8
                if ColourNoiseForge.everyStep:
                    t += 16

                self.presetList.append((name, c, s, s1, s2, t))
                self.presetList = sorted(self.presetList)

                return gr.update(choices=[i[0] for i in self.presetList])

            def delColourPreset (name):
                if name != "(None)":
                    for i in range(len(self.presetList)):
                        if self.presetList[i][0] == name:
                            del (self.presetList[i])
                            return gr.update(choices=[i[0] for i in self.presetList], value=name)

                return gr.skip()

            def saveColourPresets ():
                file = os.path.abspath(colourPresets.__file__)
                text = "presetList = [\n    " + ',\n    '.join(map(str, self.presetList)) + "\n]\n"
                with open(file, 'w') as f:
                    f.write(text)

            centreNoise.click(toggleCentre, inputs=None, outputs=centreNoise, show_progress="hidden")
            lowDNoise.click(togglelowD, inputs=None, outputs=lowDNoise, show_progress="hidden")
            sharpNoise.click(toggleSharp, inputs=None, outputs=sharpNoise, show_progress="hidden")
            targetNoise.click(toggleTarget, inputs=None, outputs=targetNoise, show_progress="hidden")
            everyStep.click(toggleEvery, inputs=None, outputs=everyStep, show_progress="hidden")
            addPreset.click(addColourPreset, inputs=[preset, initialNoise, noiseStrength, stepApplyP1, stepApplyP2], outputs=preset, show_progress="hidden")
            delPreset.click(delColourPreset, inputs=preset, outputs=preset, show_progress="hidden")
            savePreset.click(saveColourPresets, inputs=None, outputs=None, show_progress="hidden")

        self.infotext_fields = [
            (enabled,       lambda d: d.get("cn_enabled", False)),
            (noiseStrength, "cn_noiseStr"),
            (initialNoise,  "cn_noise"),
            (stepApplyP1,   "cn_stepP1"),
            (stepApplyP2,   "cn_stepP2"),
        ]

        return enabled, initialNoise, noiseStrength, stepApplyP1, stepApplyP2


    def process_before_every_sampling (self, params, *args, **kwargs):
        enabled, initialNoise, noiseStrength, stepApplyP1, stepApplyP2 = args

        if not enabled:
            return

        match len(initialNoise):
            case 7:
                initialNoiseR = int(initialNoise[1:3], 16) / 255.0
                initialNoiseG = int(initialNoise[3:5], 16) / 255.0
                initialNoiseB = int(initialNoise[5:7], 16) / 255.0
            case 4:
                initialNoiseR = int(initialNoise[1:2], 16) / 15.0
                initialNoiseG = int(initialNoise[2:3], 16) / 15.0
                initialNoiseB = int(initialNoise[3:4], 16) / 15.0
            case _:
                return

        hr_pass = params.is_hr_pass

        x = kwargs['x']
        n, c, h, w = x.size()

        ColourNoiseForge.noiseStrength = noiseStrength

        if ColourNoiseForge.centreNoise:
            for z in range(n):
                for y in range(c):
                    x[z][y] -= x[z][y].mean()

        if ColourNoiseForge.lowDNoise:
            for z in range(n): #3,5,9
                blur2 = TF.gaussian_blur(x[z], 1)
                blur4 = TF.gaussian_blur(x[z], 3)
                blur8 = TF.gaussian_blur(x[z], 7)
                x[z] = (0.0125 * blur8) + (0.0125 * blur4) + (0.05 * blur2) + (0.95 * x[z])

        #   sharpen the initial noise, using trial derived values
        if ColourNoiseForge.sharpNoise:
            minDim = 1 + 2 * (min(w, h) // 2)
            for z in range(n):
                blurred = TF.gaussian_blur(x[z], minDim)
                x[z] = 1.025*x[z] - 0.025*blurred

        if ColourNoiseForge.noiseStrength != 0.0:
            colour_image_size = (512,512)

            imageR = torch.tensor(numpy.full(colour_image_size, (initialNoiseR), dtype=numpy.float32))
            imageG = torch.tensor(numpy.full(colour_image_size, (initialNoiseG), dtype=numpy.float32))
            imageB = torch.tensor(numpy.full(colour_image_size, (initialNoiseB), dtype=numpy.float32))
            image = torch.stack((imageR, imageG, imageB), dim=0)
            image = image.unsqueeze(0)

            latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
            del imageR, imageG, imageB, image

            latent = latent[:, :, 8:-8, 8:-8]

            ColourNoiseForge.latent = latent.mean(dim=[-2, -1], keepdim=True)
        else:
            ColourNoiseForge.latent = None

        def apply_colour(self):
            if ColourNoiseForge.latent is not None:
                if (not hr_pass and self.sampling_step >= stepApplyP1 * (self.total_sampling_steps-1)) or \
                   (hr_pass and self.sampling_step >= stepApplyP2 * (self.total_sampling_steps-1)) or \
                   ColourNoiseForge.everyStep:

                    latent = ColourNoiseForge.latent
                    x = self.x

                    latent -= latent.min()
                    latent /= latent.max()
                    latent *= x.max() - x.min()
                    latent += x.min()

                    strength = ColourNoiseForge.noiseStrength

                    if ColourNoiseForge.targetted:
                        latent = latent.repeat(1, 1, x.shape[2], x.shape[3])
                        dist = (x - latent).abs()
                        dist /= dist.max()
                        latent *= (1.0 - dist) ** 9.0
                    #   method 1: mean moves toward colour
                        x += latent * strength * 5.0
                    else:
                    #   method 0: mean stays approximately the same
                        torch.lerp (x, latent, strength, out=x)

                    ColourNoiseForge.latent = None
                    self.x = x

                    if not ColourNoiseForge.everyStep or self.sampling_step == (self.total_sampling_steps-1):
                        remove_current_script_callbacks()

        on_cfg_denoiser(apply_colour)

        return


    def process(self, params, *script_args, **kwargs):
        enabled, initialNoise, noiseStrength, stepApplyP1, stepApplyP2 = script_args

        if not enabled or len(initialNoise) not in [4,7]:
            return

        params.extra_generation_params.update({
            "cn_enabled"        :   enabled,
        })
        if noiseStrength != 0:
            params.extra_generation_params.update(dict(cn_noise = initialNoise, cn_noiseStr = noiseStrength, cn_stepP1 = stepApplyP1, cn_stepP2 = stepApplyP2))
        if ColourNoiseForge.centreNoise:
            params.extra_generation_params.update({"cn_centreNoise"     : ColourNoiseForge.centreNoise,})
        if ColourNoiseForge.lowDNoise:
            params.extra_generation_params.update({"cn_lowDNoise"       : ColourNoiseForge.lowDNoise,})
        if ColourNoiseForge.sharpNoise:
            params.extra_generation_params.update({"cn_sharpNoise"      : ColourNoiseForge.sharpNoise,})
        if ColourNoiseForge.targetted:
            params.extra_generation_params.update({"cn_targettedColour" : ColourNoiseForge.targetted,})
        if ColourNoiseForge.everyStep:
            params.extra_generation_params.update({"cn_everyStep"       : ColourNoiseForge.everyStep,})

        return

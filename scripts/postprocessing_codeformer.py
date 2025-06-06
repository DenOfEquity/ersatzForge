from PIL import Image
import numpy as np

from modules import scripts_postprocessing, codeformer_model, face_restoreformer_model, ui_components
import gradio as gr


class ScriptPostprocessingCodeFormer(scripts_postprocessing.ScriptPostprocessing):
    name = "CodeFormer"
    order = 3000

    def ui(self):
        with ui_components.InputAccordion(False, label="CodeFormer") as enable:
            with gr.Row():
                codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Visibility", value=1.0, elem_id="extras_codeformer_visibility")
                codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Weight (0 = maximum effect, 1 = minimum effect)", value=0, elem_id="extras_codeformer_weight")

        return {
            "enable": enable,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, codeformer_visibility, codeformer_weight):
        if codeformer_visibility == 0 or not enable:
            return

        source_img = pp.image.convert("RGB")

        restored_img = codeformer_model.codeformer.restore(np.array(source_img, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(source_img, res, codeformer_visibility)

        pp.image = res
        pp.info["CodeFormer visibility"] = round(codeformer_visibility, 3)
        pp.info["CodeFormer weight"] = round(codeformer_weight, 3)


class ScriptPostprocessingRestoreFormer(scripts_postprocessing.ScriptPostprocessing):
    name = "RestoreFormer"
    order = 3001

    def ui(self):
        with ui_components.InputAccordion(False, label="RestoreFormer") as enable:
            with gr.Row():
                restoreformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Visibility", value=1.0, elem_id="extras_restoreformer_visibility")

        return {
            "enable": enable,
            "restoreformer_visibility": restoreformer_visibility,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, restoreformer_visibility):
        if restoreformer_visibility == 0 or not enable:
            return

        source_img = pp.image.convert("RGB")

        restored_img = face_restoreformer_model.restoreformer.restore(np.array(source_img, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if restoreformer_visibility < 1.0:
            res = Image.blend(source_img, res, restoreformer_visibility)

        pp.image = res
        pp.info["RestoreFormer visibility"] = round(restoreformer_visibility, 3)

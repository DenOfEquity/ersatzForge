import gradio
import json

from modules import scripts
from modules.infotext_utils import parse_generation_parameters, PasteField
from modules.ui_components import InputAccordion, ToolButton


class ScriptSeed(scripts.ScriptBuiltinUI):
    section = "seed"
    create_group = False

    def __init__(self):
        self.seed = None
        self.reuse_seed = None
        self.reuse_subseed = None

    def title(self):
        return "Seed"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Seed (checked: random)") as random:
            with gradio.Row():
                self.seed = gradio.Number(label="Seed", value=-1, minimum=-1, maximum=4294967295, precision=0, elem_id=self.elem_id("seed"), min_width=100)

                subseed = gradio.Number(label="Variation seed", value=-1, minimum=-1, maximum=4294967295, precision=0, scale=0, elem_id=self.elem_id("subseed"))
                subseed_strength = gradio.Number(label="Variation strength", value=0, minimum=0, maximum=1, step=0.01, scale=0, elem_id=self.elem_id("subseed_strength"))
                random_seed = ToolButton("\U0001f3b2\ufe0f", elem_id=self.elem_id("random_seed"), tooltip="set Seed (and Variation) to -1, which means random values will be used") # 🎲️
                reuse_seed = ToolButton("\u267b\ufe0f", elem_id=self.elem_id("reuse_seed"), tooltip="reuse Seed (and Variation) from last generation") # ♻️

            with gradio.Row():
                seed_resize_from_w = gradio.Number(label="Resize from width", value=0, minimum=0, maximum=2048, step=8, precision=0, elem_id=self.elem_id("seed_resize_from_w"))
                seed_resize_from_h = gradio.Number(label="Resize from height", value=0, minimum=0, maximum=2048, step=8, precision=0, elem_id=self.elem_id("seed_resize_from_h"))


        random_seed.click(fn=lambda: (-1, -1), inputs=None, outputs=[self.seed, subseed], show_progress="hidden")

        self.infotext_fields = [
            PasteField(self.seed, "Seed", api="seed"),
            PasteField(subseed, "Variation seed", api="subseed"),
            PasteField(subseed_strength, "Variation seed strength", api="subseed_strength"),
            PasteField(seed_resize_from_w, "Seed resize from-1", api="seed_resize_from_h"),
            PasteField(seed_resize_from_h, "Seed resize from-2", api="seed_resize_from_w"),
        ]

        self.on_after_component(lambda x: connect_reuse_seed(self.seed, subseed, reuse_seed, x.component), elem_id=f'generation_info_{self.tabname}')

        return random, self.seed, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h

    def setup(self, p, random, seed, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h):
        p.seed = -1 if random else seed

        if subseed_strength > 0:
            p.subseed = -1 if random else subseed
            p.subseed_strength = subseed_strength

        if seed_resize_from_w > 0 and seed_resize_from_h > 0:
            p.seed_resize_from_w = seed_resize_from_w
            p.seed_resize_from_h = seed_resize_from_h


def connect_reuse_seed(seed: gradio.Number, subseed: gradio.Number, reuse_seed: ToolButton, generation_info: gradio.Textbox):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""

    def copy_seed(gen_info_string: str, index):
        s = -1
        ss = -1
        try:
            gen_info = json.loads(gen_info_string)
            infotext = gen_info.get("infotexts")[index]
            gen_parameters = parse_generation_parameters(infotext, [])
            s = int(gen_parameters.get("Seed", -1))
            ss = int(gen_parameters.get("Variation seed", -1))
        except Exception:
            pass

        return s, ss

    reuse_seed.click(
        fn=copy_seed,
        js="(x, y) => [x, selected_gallery_index()]",
        inputs=[generation_info, seed],
        outputs=[seed, subseed],
        show_progress="hidden",
    )

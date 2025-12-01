# cut down from 'cn_in_extras_tab' extension by light-and-ray

import copy
import numpy
import gradio as gr
from PIL import Image
from modules import scripts, scripts_postprocessing, shared
from modules.ui_components import InputAccordion
from modules_forge.utils import HWC3
from lib_controlnet import global_state, external_code

forbidden_prefixes = ['inpaint', 'tile', 't2ia_style', 'revision', 'reference',
    'ip-adapter', 'instant_id_face_embedding', 'CLIP', 'InsightFace', 'facexlib', 'PuLID']

g_preprocessor_names = None
def getPreprocessorNames():
    global g_preprocessor_names
    if g_preprocessor_names is None:
        tmp_list = global_state.get_all_preprocessor_names()

        g_preprocessor_names = []
        for preprocessor in tmp_list:
            if any(preprocessor.lower().startswith(x.lower()) for x in forbidden_prefixes):
                continue
            g_preprocessor_names.append(preprocessor)

    return g_preprocessor_names


def get_default_ui_unit():
    cls = external_code.ControlNetUnit
    return cls(
        enabled=False,
        module="none",
        model="None"
    )


class CNInExtrasTab(scripts_postprocessing.ScriptPostprocessing):
    name = 'ControlNet Preprocessor'
    order = 20001

    cn_pp_cache = None
    cn_pp_hash = None

    def ui(self):
        self.default_unit = get_default_ui_unit()
        with InputAccordion(False, label='ControlNet Preprocessor') as self.enable:
            with gr.Row():
                modulesList = getPreprocessorNames()
                self.module = gr.Dropdown(modulesList, label="Module", value=modulesList[0])
                self.pixel_perfect = gr.Checkbox(
                    label="Pixel Perfect",
                    value=True,
                    elem_id="extras_controlnet_pixel_perfect_checkbox",
                )
            with gr.Row(visible=False) as self.advanced:
                self.processor_res = gr.Slider(
                    label="Preprocessor resolution",
                    value=self.default_unit.processor_res,
                    minimum=64,
                    maximum=2048,
                    visible=False,
                    interactive=True,
                    elem_id="extras_controlnet_preprocessor_resolution_slider",
                )
                self.threshold_a = gr.Slider(
                    label="Threshold A",
                    value=self.default_unit.threshold_a,
                    minimum=64,
                    maximum=1024,
                    visible=False,
                    interactive=True,
                    elem_id="extras_controlnet_threshold_A_slider",
                )
                self.threshold_b = gr.Slider(
                    label="Threshold B",
                    value=self.default_unit.threshold_b,
                    minimum=64,
                    maximum=1024,
                    visible=False,
                    interactive=True,
                    elem_id="extras_controlnet_threshold_B_slider",
                )

        self.register_build_sliders()
        args = {
            'enable': self.enable,
            'module': self.module,
            'pixel_perfect': self.pixel_perfect,
            'processor_res' : self.processor_res,
            'threshold_a' : self.threshold_a,
            'threshold_b' : self.threshold_b,
        }
        return args


    def register_build_sliders(self):
        def build_sliders(module: str, pp: bool):
            preprocessor = global_state.get_preprocessor(module)

            slider_resolution_kwargs = preprocessor.slider_resolution.gradio_update_kwargs.copy()

            if pp:
                slider_resolution_kwargs['visible'] = False

            grs = [
                gr.update(**slider_resolution_kwargs),
                gr.update(**preprocessor.slider_1.gradio_update_kwargs.copy()),
                gr.update(**preprocessor.slider_2.gradio_update_kwargs.copy()),
                gr.update(visible=True),
            ]

            return grs
        inputs = [
            self.module,
            self.pixel_perfect,
        ]
        outputs = [
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.advanced,
        ]
        self.module.change(build_sliders, inputs=inputs, outputs=outputs, show_progress="hidden")
        self.pixel_perfect.change(build_sliders, inputs=inputs, outputs=outputs, show_progress="hidden")


    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if args == {} or not args['enable']:
            return

        if "Stereogram" in [x.name for x in scripts.scripts_postproc.scripts]: # this checks if Stereogram script exists, not if enabled
            pp.original_image = pp.image.copy()

        w, h = pp.image.size
        image = HWC3(numpy.asarray(pp.image).astype(numpy.uint8))

        if args['pixel_perfect']:
            processor_res = external_code.pixel_perfect_resolution(
                image,
                target_H=h,
                target_W=w,
                resize_mode=external_code.ResizeMode.RESIZE,
            )
        else:
            processor_res = args['processor_res']

        cache_key = hash(image.tobytes()), args['module'], str(processor_res), str(args['threshold_a']), str(args['threshold_b'])
        if cache_key == self.cn_pp_hash:
            pp.image = self.cn_pp_cache
        else:
            module = global_state.get_preprocessor(args['module'])

            detected_map = module(
                input_image=image,
                resolution=processor_res,
                slider_1=args['threshold_a'],
                slider_2=args['threshold_b'],
            )

            pp.image = Image.fromarray(numpy.ascontiguousarray(detected_map.clip(0, 255).astype(numpy.uint8)).copy())

            if not (shared.state.interrupted or shared.state.skipped):
                self.cn_pp_cache = pp.image
                self.cn_pp_hash = cache_key

        info = copy.copy(args)
        del info['enable']
        if info['pixel_perfect']:
            del info['processor_res']
        pp.info['ControlNet Preprocessor'] = str(info)

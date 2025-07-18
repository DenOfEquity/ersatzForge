import json
import gradio as gr
from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import hashlib

from lib_controlnet.utils import judge_image_type
from lib_controlnet import (
    global_state,
    external_code,
)
from lib_controlnet.logging import logger
from lib_controlnet.controlnet_ui.openpose_editor import OpenposeEditor
from lib_controlnet.controlnet_ui.photopea import Photopea
from lib_controlnet.enums import InputMode, HiResFixOption
from modules import shared
from modules.ui_components import FormRow
from modules_forge.utils import HWC3
from lib_controlnet.external_code import UiControlNetUnit
from modules.ui_components import ToolButton
from gradio_rangeslider import RangeSlider
from modules_forge.forge_canvas.canvas import ForgeCanvas


@dataclass
class A1111Context:
    """Contains all components from A1111."""

    img2img_batch_input_dir = None
    img2img_batch_output_dir = None
    txt2img_submit_button = None
    img2img_submit_button = None

    # Slider controls from A1111 WebUI.
    txt2img_w_slider = None
    txt2img_h_slider = None
    img2img_w_slider = None
    img2img_h_slider = None

    @property
    def ui_initialized(self) -> bool:
        optional_components = {
            # Optional components are only available after A1111 v1.7.0.
        }
        return all(
            c
            for name, c in vars(self).items()
            if name not in optional_components.values()
        )

    def set_component(self, component):
        id_mapping = {
            "img2img_batch_input_dir": "img2img_batch_input_dir",
            "img2img_batch_output_dir": "img2img_batch_output_dir",
            "txt2img_generate": "txt2img_submit_button",
            "img2img_generate": "img2img_submit_button",
            "txt2img_width": "txt2img_w_slider",
            "txt2img_height": "txt2img_h_slider",
            "img2img_width": "img2img_w_slider",
            "img2img_height": "img2img_h_slider",
        }
        elem_id = getattr(component, "elem_id", None)
        # Do not set component if it has already been set.
        # https://github.com/Mikubill/sd-webui-controlnet/issues/2587
        if elem_id in id_mapping and getattr(self, id_mapping[elem_id]) is None:
            setattr(self, id_mapping[elem_id], component)
            logger.debug(f"Setting {elem_id}.")
            logger.debug(
                f"A1111 initialized {sum(c is not None for c in vars(self).values())}/{len(vars(self).keys())}."
            )


class ControlNetUiGroup(object):
    refresh_symbol = "\U0001f504"  # 🔄
    switch_values_symbol = "\U000021C5"  # ⇅
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝

    tooltips = {
        "🔄": "Refresh",
        "\u2934": "Send dimensions to stable diffusion",
        "💥": "Run preprocessor",
        "📝": "Open new canvas",
        "📷": "Enable webcam",
        "⇄": "Mirror webcam",
    }

    global_batch_input_dir = gr.Textbox(
        label="Controlnet input directory",
        placeholder="Leave empty to use input directory",
        **shared.hide_dirs,
        elem_id="controlnet_batch_input_dir",
    )
    a1111_context = A1111Context()
    # All ControlNetUiGroup instances created.
    all_ui_groups: List["ControlNetUiGroup"] = []

    @property
    def width_slider(self):
        if self.is_img2img:
            return ControlNetUiGroup.a1111_context.img2img_w_slider
        else:
            return ControlNetUiGroup.a1111_context.txt2img_w_slider

    @property
    def height_slider(self):
        if self.is_img2img:
            return ControlNetUiGroup.a1111_context.img2img_h_slider
        else:
            return ControlNetUiGroup.a1111_context.txt2img_h_slider

    def __init__(
        self,
        is_img2img: bool,
        default_unit: external_code.ControlNetUnit,
        photopea: Optional[Photopea] = None,
    ):
        # Whether callbacks have been registered.
        self.callbacks_registered: bool = False
        # Whether the render method on this object has been called.
        self.ui_initialized: bool = False

        self.is_img2img = is_img2img
        self.default_unit = default_unit
        self.photopea = photopea
        self.webcam_enabled = False
        self.webcam_mirrored = False

        # Note: All gradio elements declared in `render` will be defined as member variable.
        # Update counter to trigger a force update of UiControlNetUnit.
        # dummy_gradio_update_trigger is useful when a field with no event subscriber available changes.
        # e.g. gr.Gallery, gr.State, etc. After an update to gr.State / gr.Gallery, please increment
        # this counter to trigger a sync update of UiControlNetUnit.
        self.dummy_gradio_update_trigger = None
        self.enabled = None
        self.upload_tab = None
        self.image = None
        self.generated_image_group = None
        self.generated_image = None
        self.mask_image_group = None
        self.mask_image = None
        self.batch_tab = None
        self.batch_image_dir = None
        self.merge_tab = None
        self.batch_input_gallery = None
        self.batch_mask_gallery = None
        self.create_canvas = None
        self.canvas_width = None
        self.canvas_height = None
        self.canvas_create_button = None
        self.canvas_cancel_button = None
        self.open_new_canvas_button = None
        self.send_dimen_button = None
        self.pixel_perfect = None
        self.preprocessor_preview = None
        self.mask_upload = None
        self.type_filter = None
        self.module = None
        self.trigger_preprocessor = None
        self.model = None
        self.refresh_models = None
        self.weight = None
        self.timestep_range = None
        self.guidance_start = None
        self.guidance_end = None
        self.advanced = None
        self.processor_res = None
        self.threshold_a = None
        self.threshold_b = None
        self.control_mode = None
        self.resize_mode = None
        self.use_preview_as_input = None
        self.openpose_editor = None
        self.upload_independent_img_in_img2img = None
        self.image_upload_panel = None
        self.input_mode = gr.State(InputMode.SIMPLE)
        self.hr_option = None
        self.batch_image_dir_state = None
        self.output_dir_state = None

        # Internal states for UI state pasting.
        self.prevent_next_n_module_update = 0
        self.prevent_next_n_slider_value_update = 0

        ControlNetUiGroup.all_ui_groups.append(self)

    def render(self, tabname: str, elem_id_tabname: str) -> None:
        """The pure HTML structure of a single ControlNetUnit. Calling this
        function will populate `self` with all gradio element declared
        in local scope.

        Args:
            tabname:
            elem_id_tabname:

        Returns:
            None
        """
        self.dummy_gradio_update_trigger = gr.Number(value=0, visible=False)
        self.openpose_editor = OpenposeEditor()

        with gr.Group(visible=not self.is_img2img) as self.image_upload_panel:
            with gr.Tabs(visible=True):
                with gr.Tab(label="Single image") as self.upload_tab:
                    with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                        with gr.Group(elem_classes=["cnet-input-image-group"]):
                            self.image = ForgeCanvas(
                                elem_id=f"{elem_id_tabname}_{tabname}_input_image",
                                elem_classes=["cnet-image"],
                                contrast_scribbles=True,
                                height=300,
                                numpy=True
                            )
                            self.openpose_editor.render_upload()

                        with gr.Group(
                                visible=False, elem_classes=["cnet-generated-image-group"]
                        ) as self.generated_image_group:
                            self.generated_image = ForgeCanvas(
                                elem_id=f"{elem_id_tabname}_{tabname}_generated_image",
                                elem_classes=["cnet-image"],
                                height=300,
                                no_scribbles=True,
                                no_upload=True,
                                numpy=True
                            )

                            with gr.Group(
                                    elem_classes=["cnet-generated-image-control-group"]
                            ):
                                if self.photopea:
                                    self.photopea.render_child_trigger()
                                self.openpose_editor.render_edit()

                        with gr.Group(
                                visible=False, elem_classes=["cnet-mask-image-group"]
                        ) as self.mask_image_group:
                            self.mask_image = ForgeCanvas(
                                elem_id=f"{elem_id_tabname}_{tabname}_mask_image",
                                elem_classes=["cnet-mask-image"],
                                height=300,
                                scribble_color='#FFFFFF',
                                scribble_width=1,
                                scribble_alpha_fixed=True,
                                scribble_color_fixed=True,
                                scribble_softness_fixed=True,
                                numpy=True
                            )

                with gr.Tab(label="Batch folder") as self.batch_tab:
                    with gr.Row():
                        self.batch_image_dir = gr.Textbox(
                            label="Input directory",
                            placeholder="Input directory path to the control images.",
                            elem_id=f"{elem_id_tabname}_{tabname}_batch_image_dir",
                        )
                        self.batch_mask_dir = gr.Textbox(
                            label="Mask directory",
                            placeholder="Mask directory path to the control images.",
                            elem_id=f"{elem_id_tabname}_{tabname}_batch_mask_dir",
                            visible=False,
                        )

                with gr.Tab(label="Multiple images") as self.merge_tab:
                    with gr.Row():
                        with gr.Column():
                            self.batch_input_gallery = gr.Gallery(
                                columns=[4], rows=[2], object_fit="contain", height="auto", label="Images"
                            )
                        with gr.Group(visible=False, elem_classes=["cnet-mask-gallery-group"]) as self.batch_mask_gallery_group:
                            with gr.Column():
                                self.batch_mask_gallery = gr.Gallery(
                                    columns=[4], rows=[2], object_fit="contain", height="auto", label="Masks"
                                )

            self.upload_tab.select(fn=lambda: InputMode.SIMPLE, inputs=None, outputs=[self.input_mode], show_progress=False)
            self.batch_tab.select(fn=lambda: InputMode.BATCH, inputs=None, outputs=[self.input_mode], show_progress=False)
            self.merge_tab.select(fn=lambda: InputMode.MERGE, inputs=None, outputs=[self.input_mode], show_progress=False)

            if self.photopea:
                self.photopea.attach_photopea_output(self.generated_image.background)

            with gr.Accordion(
                label="Open New Canvas", visible=False
            ) as self.create_canvas:
                self.canvas_width = gr.Slider(
                    label="New Canvas Width",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_width",
                )
                self.canvas_height = gr.Slider(
                    label="New Canvas Height",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_height",
                )
                with gr.Row():
                    self.canvas_create_button = gr.Button(
                        value="Create New Canvas",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_create_button",
                    )
                    self.canvas_cancel_button = gr.Button(
                        value="Cancel",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_cancel_button",
                    )

            with gr.Row(elem_classes="controlnet_image_controls"):
                gr.HTML(
                    value="<p>Set the preprocessor to [invert] If your image has white background and black lines.</p>",
                    elem_classes="controlnet_invert_warning",
                )
                self.open_new_canvas_button = ToolButton(
                    value=ControlNetUiGroup.open_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button",
                    elem_classes=["cnet-toolbutton"],
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.open_symbol],
                )
                self.send_dimen_button = ToolButton(
                    value=ControlNetUiGroup.tossup_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_dimen_button",
                    elem_classes=["cnet-toolbutton"],
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossup_symbol],
                )

        with FormRow(elem_classes=["controlnet_main_options"]):
            self.enabled = gr.Checkbox(
                label="Enable",
                value=self.default_unit.enabled,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_enable_checkbox",
                elem_classes=["cnet-unit-enabled"],
            )
            self.pixel_perfect = gr.Checkbox(
                label="Pixel perfect",
                value=self.default_unit.pixel_perfect,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
            )
            self.preprocessor_preview = gr.Checkbox(
                label="Allow preview",
                value=False,
                elem_classes=["cnet-allow-preview"],
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox",
                visible=not self.is_img2img,
            )
            self.mask_upload = gr.Checkbox(
                label="Use mask",
                value=False,
                elem_classes=["cnet-mask-upload"],
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_mask_upload_checkbox",
                visible=not self.is_img2img,
            )
            self.use_preview_as_input = gr.State(value=False)

            if self.is_img2img:
                self.upload_independent_img_in_img2img = gr.Checkbox(
                    label="Independent control",
                    value=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_same_img2img_checkbox",
                    elem_classes=["cnet-unit-same_img2img"],
                )
            else:
                self.upload_independent_img_in_img2img = None

        self.type_filter = gr.Radio(
            global_state.get_all_preprocessor_tags(),
            label="Control type",
            value="All",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_type_filter_radio",
            elem_classes="controlnet_control_type_filter_group",
            scale=0,
        )

        with gr.Row(elem_classes=["controlnet_preprocessor_model", "controlnet_row"]):
            self.module = gr.Dropdown(
                global_state.get_all_preprocessor_names(),
                label="Preprocessor",
                value=self.default_unit.module,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
            )
            self.trigger_preprocessor = ToolButton(
                value=ControlNetUiGroup.trigger_symbol,
                visible=not self.is_img2img,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor",
                elem_classes=["cnet-run-preprocessor", "cnet-toolbutton"],
                tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.trigger_symbol],
            )
            self.model = gr.Dropdown(
                global_state.get_all_controlnet_names(),
                label="Model",
                value=self.default_unit.model,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
            )
            self.refresh_models = ToolButton(
                value=ControlNetUiGroup.refresh_symbol,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_refresh_models",
                elem_classes=["cnet-toolbutton"],
                tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.refresh_symbol],
            )

        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            self.weight = gr.Slider(
                label="Control Weight",
                value=self.default_unit.weight,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_weight_slider",
                elem_classes="controlnet_control_weight_slider",
            )
            self.timestep_range = RangeSlider(
                label='Timestep Range',
                minimum=0,
                maximum=1.0,
                value=(self.default_unit.guidance_start, self.default_unit.guidance_end),
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_step_slider",
                elem_classes="controlnet_control_step_slider",
            )
            self.guidance_start = gr.State(self.default_unit.guidance_start)
            self.guidance_end = gr.State(self.default_unit.guidance_end)

        with FormRow(elem_classes=["controlnet_control_type", "controlnet_row"]):
            self.control_mode = gr.Dropdown(
                choices=[e.value for e in external_code.ControlMode],
                value=self.default_unit.control_mode.value,
                label="Control mode",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_mode_radio",
                elem_classes="controlnet_control_mode_radio",
            )
            self.resize_mode = gr.Dropdown(
                choices=[e.value for e in external_code.ResizeMode],
                value=self.default_unit.resize_mode.value,
                label="Resize mode",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio",
                elem_classes="controlnet_resize_mode_radio",
                visible=not self.is_img2img,
            )
            self.hr_option = gr.Dropdown(
                choices=[e.value for e in HiResFixOption],
                value=self.default_unit.hr_option.value,
                label="HiRes-fix option",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_hr_option_radio",
                elem_classes="controlnet_hr_option_radio",
                visible=not self.is_img2img,
            )

        self.timestep_range.change(
            lambda x: (x[0], x[1]),
            inputs=[self.timestep_range],
            outputs=[self.guidance_start, self.guidance_end]
        )

        # advanced options
        with gr.Column(visible=False) as self.advanced:
            self.processor_res = gr.Slider(
                label="Preprocessor resolution",
                value=self.default_unit.processor_res,
                minimum=64,
                maximum=2048,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider",
            )
            self.threshold_a = gr.Slider(
                label="Threshold A",
                value=self.default_unit.threshold_a,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider",
            )
            self.threshold_b = gr.Slider(
                label="Threshold B",
                value=self.default_unit.threshold_b,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider",
            )

        self.batch_image_dir_state = gr.State("")
        self.output_dir_state = gr.State("")
        unit_args = (
            self.input_mode,
            self.use_preview_as_input,
            self.batch_image_dir,
            self.batch_mask_dir,
            self.batch_input_gallery,
            self.batch_mask_gallery,
            self.generated_image.background,
            self.mask_image.background,
            self.mask_image.foreground,
            self.hr_option,
            self.enabled,
            self.module,
            self.model,
            self.weight,
            self.image.background,
            self.image.foreground,
            self.resize_mode,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.guidance_start,
            self.guidance_end,
            self.pixel_perfect,
            self.control_mode,
        )

        unit = gr.State(self.default_unit)
        for comp in unit_args + (self.dummy_gradio_update_trigger,):
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)

            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit
                )

        (
            ControlNetUiGroup.a1111_context.img2img_submit_button
            if self.is_img2img
            else ControlNetUiGroup.a1111_context.txt2img_submit_button
        ).click(
            fn=UiControlNetUnit,
            inputs=list(unit_args),
            outputs=unit,
            queue=False,
        )
        self.register_core_callbacks()
        self.ui_initialized = True
        return unit

    def register_send_dimensions(self):
        """Register event handler for send dimension button."""

        def send_dimensions(image):
            def closesteight(num):
                rem = num % 8
                if rem <= 4:
                    return round(num - rem)
                else:
                    return round(num + (8 - rem))

            if image is not None:
                return closesteight(image.shape[1]), closesteight(image.shape[0])
            else:
                return gr.Slider.update(), gr.Slider.update()

        self.send_dimen_button.click(
            fn=send_dimensions,
            inputs=[self.image.background],
            outputs=[self.width_slider, self.height_slider],
            show_progress=False,
        )

    def register_refresh_all_models(self):
        def refresh_all_models():
            global_state.update_controlnet_filenames()
            return gr.Dropdown.update(
                choices=global_state.get_all_controlnet_names(),
            )

        self.refresh_models.click(
            refresh_all_models,
            outputs=[self.model],
            show_progress=False,
        )

    def register_build_sliders(self):
        def build_sliders(module: str, pp: bool):

            logger.debug(
                f"Prevent update slider value: {self.prevent_next_n_slider_value_update}"
            )
            logger.debug(f"Build slider for module: {module} - {pp}")

            preprocessor = global_state.get_preprocessor(module)

            slider_resolution_kwargs = preprocessor.slider_resolution.gradio_update_kwargs.copy()

            if pp:
                slider_resolution_kwargs['visible'] = False

            grs = [
                gr.update(**slider_resolution_kwargs),
                gr.update(**preprocessor.slider_1.gradio_update_kwargs.copy()),
                gr.update(**preprocessor.slider_2.gradio_update_kwargs.copy()),
                gr.update(visible=True),
                gr.update(visible=not preprocessor.do_not_need_model),
                gr.update(visible=not preprocessor.do_not_need_model),
                gr.update(interactive=preprocessor.show_control_mode),
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
            self.model,
            self.refresh_models,
            self.control_mode,
        ]
        self.module.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )
        self.pixel_perfect.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )

        def filter_selected(k: str):
            logger.debug(f"Prevent update {self.prevent_next_n_module_update}")
            logger.debug(f"Switch to control type {k}")

            filtered_preprocessor_list = global_state.get_filtered_preprocessor_names(k)
            filtered_controlnet_names = global_state.get_filtered_controlnet_names(k)
            default_preprocessor = filtered_preprocessor_list[0]
            default_controlnet_name = filtered_controlnet_names[0]

            if k != 'All':
                if len(filtered_preprocessor_list) > 1:
                    default_preprocessor = filtered_preprocessor_list[1]
                if len(filtered_controlnet_names) > 1:
                    default_controlnet_name = filtered_controlnet_names[1]

            if self.prevent_next_n_module_update > 0:
                self.prevent_next_n_module_update -= 1
                return [
                    gr.Dropdown.update(choices=filtered_preprocessor_list),
                    gr.Dropdown.update(choices=filtered_controlnet_names),
                ]
            else:
                return [
                    gr.Dropdown.update(
                        value=default_preprocessor, choices=filtered_preprocessor_list
                    ),
                    gr.Dropdown.update(
                        value=default_controlnet_name, choices=filtered_controlnet_names
                    ),
                ]

        self.type_filter.change(
            fn=filter_selected,
            inputs=[self.type_filter],
            outputs=[self.module, self.model],
            show_progress=False,
        )

    def register_run_annotator(self):
        def run_annotator(image, mask, module, pres, pthr_a, pthr_b, t2i_w, t2i_h, pp, rm):
            if image is None:
                return (
                    gr.update(visible=True),
                    None,
                    gr.update(),
                    *self.openpose_editor.update(""),
                )

            img = HWC3(image)
            mask = HWC3(mask)

            if not (mask > 5).any():
                mask = None

            preprocessor = global_state.get_preprocessor(module)

            if pp:
                pres = external_code.pixel_perfect_resolution(
                    img,
                    target_H=t2i_h,
                    target_W=t2i_w,
                    resize_mode=external_code.resize_mode_from_value(rm),
                )

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = ""

                def accept(self, json_dict: dict) -> None:
                    self.value = json.dumps(json_dict)

            json_acceptor = JsonAcceptor()

            def is_openpose(module: str):
                return "openpose" in module

            # Only openpose preprocessor returns a JSON output, pass json_acceptor
            # only when a JSON output is expected. This will make preprocessor cache
            # work for all other preprocessors other than openpose ones. JSON acceptor
            # instance are different every call, which means cache will never take
            # effect.
            # TODO: Maybe we should let `preprocessor` return a Dict to alleviate this issue?
            # This requires changing all callsites though.

            # the cache referred to in the preceding paragraph is the preview image
            # the new additional caching implemented below must be enabled per preprocessor by adding 'cache' and 'cacheHash' attributes
            # (could be added to the base Preprocessor class, but can't apply to some preprocessors so seems better not to)
            # applies on preview, if a cached result was already saved on Generate
            # and saves a cached result anyway for the case where preview is generated then removed but settings are unchanged, so the cache is still valid
            # equivalent code has been added to 'process_unit_after_click_generate' in controlnet.py
            cacheAvailable = hasattr(preprocessor, "cache") and hasattr(preprocessor, "cacheHash")
            usedCache = False
            preprocessorHash = None

            if cacheAvailable:
                hash_sha256 = hashlib.sha256()
                simpleHash = str(img) + str(module) + str(pres) + str(pthr_a) + str(pthr_b)
                hash_sha256.update(simpleHash.encode('utf-8'))
                preprocessorHash = hash_sha256.hexdigest()

                if preprocessor.cache is not None and preprocessor.cacheHash == preprocessorHash:
                    logger.info(f"Preview Resolution = (cached) {pres}")
                    result = preprocessor.cache
                    usedCache = True

            if not usedCache:
                logger.info(f"Preview Resolution = {pres}")
                result = preprocessor(
                    input_image=img,
                    resolution=pres,
                    slider_1=pthr_a,
                    slider_2=pthr_b,
                    input_mask=mask,
                    json_pose_callback=json_acceptor.accept
                    if is_openpose(module)
                    else None,
                )
                if cacheAvailable:
                    preprocessor.cache = result
                    preprocessor.cacheHash = preprocessorHash

            is_image = judge_image_type(result)

            if not is_image:
                result = img

            result = external_code.visualize_inpaint_mask(result)
            return (
                gr.update(visible=True),
                result,
                # preprocessor_preview
                gr.update(value=True),
                # openpose editor
                *self.openpose_editor.update(json_acceptor.value),
            )

        self.trigger_preprocessor.click(
            fn=run_annotator,
            inputs=[
                self.image.background,
                self.image.foreground,
                self.module,
                self.processor_res,
                self.threshold_a,
                self.threshold_b,
                self.width_slider,
                self.height_slider,
                self.pixel_perfect,
                self.resize_mode,
            ],
            outputs=[
                self.generated_image.block,
                self.generated_image.background,
                self.preprocessor_preview,
                *self.openpose_editor.outputs(),
            ],
        )

    def register_shift_preview(self):
        def shift_preview(is_on):
            return (
                # generated_image
                gr.update() if is_on else gr.update(value=None),
                # generated_image_group
                gr.update(visible=is_on),
                # use_preview_as_input,
                gr.update(value=is_on),  # Now this is automatically managed ??
                # download_pose_link
                gr.update() if is_on else gr.update(value=None),
                # modal edit button
                gr.update() if is_on else gr.update(visible=False),
            )

        self.preprocessor_preview.change(
            fn=shift_preview,
            inputs=[self.preprocessor_preview],
            outputs=[
                self.generated_image.background,
                self.generated_image_group,
                self.use_preview_as_input,
                self.openpose_editor.download_link,
                self.openpose_editor.modal,
            ],
            show_progress=False,
        )

    def register_create_canvas(self):
        self.open_new_canvas_button.click(
            lambda: gr.update(visible=True),
            inputs=None,
            outputs=self.create_canvas,
            show_progress=False,
        )
        self.canvas_cancel_button.click(
            lambda: gr.update(visible=False),
            inputs=None,
            outputs=self.create_canvas,
            show_progress=False,
        )

        def fn_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8), gr.update(
                visible=False
            )

        self.canvas_create_button.click(
            fn=fn_canvas,
            inputs=[self.canvas_height, self.canvas_width],
            outputs=[self.image.background, self.create_canvas],
            show_progress=False,
        )

    def register_img2img_same_input(self):
        def fn_same_checked(x):
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=False, visible=x),
            ] + [gr.update(visible=x)] * 3

        self.upload_independent_img_in_img2img.change(
            fn_same_checked,
            inputs=self.upload_independent_img_in_img2img,
            outputs=[
                self.image.background,
                self.batch_image_dir,
                self.preprocessor_preview,
                self.image_upload_panel,
                self.trigger_preprocessor,
                self.resize_mode,
            ],
            show_progress=False,
        )

    def register_shift_crop_input_image(self):
        return

    def register_shift_upload_mask(self):
        """Controls whether the upload mask input should be visible."""
        def on_checkbox_click(checked: bool, canvas_height: int, canvas_width: int):
            if not checked:
                # Clear mask_image if unchecked.
                return gr.update(visible=False), gr.update(value=None), gr.update(value=None, visible=False), \
                        gr.update(visible=False), gr.update(value=None)
            else:
                # Init an empty canvas the same size as the generation target.
                empty_canvas = np.zeros(shape=(canvas_height, canvas_width, 3), dtype=np.uint8)
                return gr.update(visible=True), gr.update(value=empty_canvas), gr.update(visible=True), \
                        gr.update(visible=True), gr.update()

        self.mask_upload.change(
            fn=on_checkbox_click,
            inputs=[self.mask_upload, self.height_slider, self.width_slider],
            outputs=[self.mask_image_group, self.mask_image.background, self.batch_mask_dir,
                     self.batch_mask_gallery_group, self.batch_mask_gallery],
            show_progress=False,
        )

        if self.upload_independent_img_in_img2img is not None:
            self.upload_independent_img_in_img2img.change(
                fn=lambda checked: (
                    # Uncheck `upload_mask` when not using independent input.
                    gr.update(visible=False, value=False)
                    if not checked
                    else gr.update(visible=True)
                ),
                inputs=[self.upload_independent_img_in_img2img],
                outputs=[self.mask_upload],
                show_progress=False,
            )

    def register_sync_batch_dir(self):
        def determine_batch_dir(batch_dir, fallback_dir, fallback_fallback_dir):
            if batch_dir:
                return batch_dir
            elif fallback_dir:
                return fallback_dir
            else:
                return fallback_fallback_dir

        batch_dirs = [
            self.batch_image_dir,
            ControlNetUiGroup.global_batch_input_dir,
            ControlNetUiGroup.a1111_context.img2img_batch_input_dir,
        ]
        for batch_dir_comp in batch_dirs:
            subscriber = getattr(batch_dir_comp, "blur", None)
            if subscriber is None:
                continue
            subscriber(
                fn=determine_batch_dir,
                inputs=batch_dirs,
                outputs=[self.batch_image_dir_state],
                queue=False,
            )

        ControlNetUiGroup.a1111_context.img2img_batch_output_dir.blur(
            fn=lambda a: a,
            inputs=[ControlNetUiGroup.a1111_context.img2img_batch_output_dir],
            outputs=[self.output_dir_state],
            queue=False,
        )

    def register_clear_preview(self):
        def clear_preview(x):
            if x:
                logger.info("Preview as input is cancelled.")
            return gr.update(value=False), gr.update(value=None)

        for comp in (
            self.pixel_perfect,
            self.module,
            self.image.background,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.upload_independent_img_in_img2img,
        ):
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)
            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=clear_preview,
                    inputs=self.use_preview_as_input,
                    outputs=[self.use_preview_as_input, self.generated_image.background],
                    show_progress=False
                )

    def register_core_callbacks(self):
        """Register core callbacks that only involves gradio components defined
        within this ui group."""
        self.register_refresh_all_models()
        self.register_build_sliders()
        self.register_shift_preview()
        self.register_create_canvas()
        self.register_clear_preview()
        self.openpose_editor.register_callbacks(
            self.generated_image,
            self.use_preview_as_input,
            self.model,
        )
        assert self.type_filter is not None
        if self.is_img2img:
            self.register_img2img_same_input()

    def register_callbacks(self):
        """Register callbacks that involves A1111 context gradio components."""
        # Prevent infinite recursion.
        if self.callbacks_registered:
            return

        self.callbacks_registered = True
        self.register_send_dimensions()
        self.register_run_annotator()
        self.register_sync_batch_dir()
        self.register_shift_upload_mask()
        if self.is_img2img:
            self.register_shift_crop_input_image()


    @staticmethod
    def reset():
        ControlNetUiGroup.a1111_context = A1111Context()
        ControlNetUiGroup.all_ui_groups = []

    @staticmethod
    def try_register_all_callbacks():
        unit_count = shared.opts.data.get("control_net_unit_count", 3)
        all_unit_count = unit_count * 2  # txt2img + img2img.
        if (
            # All A1111 components ControlNet units care about are all registered.
            ControlNetUiGroup.a1111_context.ui_initialized
            and all_unit_count == len(ControlNetUiGroup.all_ui_groups)
            and all(
                g.ui_initialized and (not g.callbacks_registered)
                for g in ControlNetUiGroup.all_ui_groups
            )
        ):
            for ui_group in ControlNetUiGroup.all_ui_groups:
                ui_group.register_callbacks()

            logger.info("ControlNet UI callback registered.")

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        if getattr(component, "elem_id", None) == "img2img_batch_inpaint_mask_dir":
            ControlNetUiGroup.global_batch_input_dir.render()
            return

        ControlNetUiGroup.a1111_context.set_component(component)

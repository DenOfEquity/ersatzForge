from __future__ import annotations
import base64
import io
import json
import os
import re
import sys

import gradio as gr
from modules.paths import data_path
from modules import shared, ui_tempdir, script_callbacks, processing, infotext_versions, images, prompt_parser, errors
from PIL import Image

from modules_forge import main_entry

sys.modules['modules.generation_parameters_copypaste'] = sys.modules[__name__]  # alias for old name

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$")
type_of_gr_update = type(gr.update())


class ParamBinding:
    def __init__(self, paste_button, tabname, source_text_component=None, source_image_component=None, source_tabname=None, override_settings_component=None, paste_field_names=None):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names or []


class PasteField(tuple):
    def __new__(cls, component, target, *, api=None):
        return super().__new__(cls, (component, target))

    def __init__(self, component, target, *, api=None):
        super().__init__()

        self.api = api
        self.component = component
        self.label = target if isinstance(target, str) else None
        self.function = target if callable(target) else None


paste_fields: dict[str, dict] = {}
registered_param_bindings: list[ParamBinding] = []


def reset():
    paste_fields.clear()
    registered_param_bindings.clear()


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


def image_from_url_text(filedata):
    if filedata is None:
        return None

    if isinstance(filedata, list):
        if len(filedata) == 0:
            return None

        filedata = filedata[0]

    if isinstance(filedata, dict) and filedata.get("is_file", False):
        filedata = filedata

    filename = None
    if type(filedata) == dict and filedata.get("is_file", False):
        filename = filedata["name"]

    elif isinstance(filedata, tuple) and len(filedata) == 2:  # gradio 4.16 sends images from gallery as a list of tuples
        return filedata[0]

    if filename:
        is_in_right_dir = ui_tempdir.check_tmp_file(shared.demo, filename)
        assert is_in_right_dir, 'trying to open image file outside of allowed directories'

        filename = filename.rsplit('?', 1)[0]
        return images.read(filename)

    if isinstance(filedata, str):
        if filedata.startswith("data:image/png;base64,"):
            filedata = filedata[len("data:image/png;base64,"):]

        filedata = base64.decodebytes(filedata.encode('utf-8'))
        image = images.read(io.BytesIO(filedata))
        return image

    return None


def add_paste_fields(tabname, init_img, fields, override_settings_component=None):

    if fields:
        for i in range(len(fields)):
            if not isinstance(fields[i], PasteField):
                fields[i] = PasteField(*fields[i])

    paste_fields[tabname] = {"init_img": init_img, "fields": fields, "override_settings_component": override_settings_component}

    # backwards compatibility for existing extensions
    import modules.ui
    if tabname == 'txt2img':
        modules.ui.txt2img_paste_fields = fields
    elif tabname == 'img2img':
        modules.ui.img2img_paste_fields = fields


def create_buttons(tabs_list):
    buttons = {}
    for tab in tabs_list:
        if tab == 'inpaint':    # backcompat
            continue
        buttons[tab] = gr.Button(f"Send to {tab}", elem_id=f"{tab}_tab")
    return buttons


def bind_buttons(buttons, send_image, send_generate_info):
    """old function for backwards compatibility; do not use this, use register_paste_params_button"""
    for tabname, button in buttons.items():
        source_text_component = send_generate_info if isinstance(send_generate_info, gr.components.Component) else None
        source_tabname = send_generate_info if isinstance(send_generate_info, str) else None

        register_paste_params_button(ParamBinding(paste_button=button, tabname=tabname, source_text_component=source_text_component, source_image_component=send_image, source_tabname=source_tabname))


def register_paste_params_button(binding: ParamBinding):
    registered_param_bindings.append(binding)


def connect_paste_params_buttons():
    for binding in registered_param_bindings:
        if binding.tabname == 'inpaint':    # backcompat
            continue

        destination_image_component = paste_fields[binding.tabname]["init_img"]
        fields = paste_fields[binding.tabname]["fields"]
        override_settings_component = binding.override_settings_component or paste_fields[binding.tabname]["override_settings_component"]

        destination_width_component = next(iter([field for field, name in fields if name == "Size-1"] if fields else []), None)
        destination_height_component = next(iter([field for field, name in fields if name == "Size-2"] if fields else []), None)

        if binding.source_image_component and destination_image_component:
            need_send_dementions = destination_width_component and binding.tabname != 'inpaint'
            if isinstance(binding.source_image_component, gr.Gallery):
                func = send_image_and_dimensions if need_send_dementions else image_from_url_text
                jsfunc = "extract_image_from_gallery"
            else:
                func = send_image_and_dimensions if need_send_dementions else lambda x: x
                jsfunc = None

            binding.paste_button.click(
                fn=func,
                _js=jsfunc,
                inputs=[binding.source_image_component],
                outputs=[destination_image_component, destination_width_component, destination_height_component] if need_send_dementions else [destination_image_component],
                show_progress=False,
            )

        if binding.source_text_component is not None and fields is not None:
            connect_paste(binding.paste_button, fields, binding.source_text_component, override_settings_component, binding.tabname)

        if binding.source_tabname is not None and fields is not None:
            paste_field_names = ['Prompt', 'Negative prompt', 'Steps', 'Face restoration'] + (["Seed"] if shared.opts.send_seed else []) + binding.paste_field_names
            binding.paste_button.click(
                fn=lambda *x: x,
                inputs=[field for field, name in paste_fields[binding.source_tabname]["fields"] if name in paste_field_names],
                outputs=[field for field, name in fields if name in paste_field_names],
                show_progress=False,
            )

        binding.paste_button.click(
            fn=None,
            _js=f"switch_to_{binding.tabname}",
            inputs=None,
            outputs=None,
            show_progress=False,
        )


def send_image_and_dimensions(x):
    if isinstance(x, Image.Image):
        img = x
    elif isinstance(x, list) and isinstance(x[0], tuple):
        img = x[0][0]
    else:
        img = image_from_url_text(x)

    if shared.opts.send_size and isinstance(img, Image.Image):
        w = img.width
        h = img.height
    else:
        w = gr.update()
        h = gr.update()

    return img, w, h


def restore_old_hires_fix_params(res):
    """for infotexts that specify old First pass size parameter, convert it into
    width, height, and hr scale"""

    firstpass_width = res.get('First pass size-1', None)
    firstpass_height = res.get('First pass size-2', None)

    if firstpass_width is None or firstpass_height is None:
        return

    firstpass_width, firstpass_height = int(firstpass_width), int(firstpass_height)
    width = int(res.get("Size-1", 512))
    height = int(res.get("Size-2", 512))

    if firstpass_width == 0 or firstpass_height == 0:
        firstpass_width, firstpass_height = processing.old_hires_fix_first_pass_dimensions(width, height)

    res['Size-1'] = firstpass_width
    res['Size-2'] = firstpass_height
    res['Hires resize-1'] = width
    res['Hires resize-2'] = height


def parse_generation_parameters(x: str, skip_fields: list[str] | None = None):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """
    if skip_fields is None:
        skip_fields = shared.opts.infotext_skip_pasting

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    if 'Civitai' in lastline and 'FLUX' in lastline:
        # Civitai really like to add random Clip skip to Flux metadata, where Clip skip is not a thing.
        lastline = lastline.replace('Clip skip: 0, ', '')
        lastline = lastline.replace('Clip skip: 1, ', '')
        lastline = lastline.replace('Clip skip: 2, ', '')
        lastline = lastline.replace('Clip skip: 3, ', '')
        lastline = lastline.replace('Clip skip: 4, ', '')
        lastline = lastline.replace('Clip skip: 5, ', '')
        lastline = lastline.replace('Clip skip: 6, ', '')
        lastline = lastline.replace('Clip skip: 7, ', '')
        lastline = lastline.replace('Clip skip: 8, ', '')

        # Civitai also add Sampler: Undefined
        lastline = lastline.replace('Sampler: Undefined, ', 'Sampler: Euler, Schedule type: Simple, ')  # <- by lllyasviel, seem to give similar results to Civitai "Undefined" Sampler

        # Civitai also confuse CFG scale and Distilled CFG Scale
        lastline = lastline.replace('CFG scale: ', 'CFG scale: 1, Distilled CFG Scale: ')

        print('Applied Forge Fix to broken Civitai Flux Meta.')

    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            m = re_imagesize.match(v)
            if m is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
            print(f"Error parsing \"{k}: {v}\"")

    # Extract styles from prompt
    if shared.opts.infotext_styles != "Ignore":
        found_styles, prompt_no_styles, negative_prompt_no_styles = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)

        same_hr_styles = True
        if ("Hires prompt" in res or "Hires negative prompt" in res) and (infotext_ver > infotext_versions.v180_hr_styles if (infotext_ver := infotext_versions.parse_version(res.get("Version"))) else True):
            hr_prompt, hr_negative_prompt = res.get("Hires prompt", prompt), res.get("Hires negative prompt", negative_prompt)
            hr_found_styles, hr_prompt_no_styles, hr_negative_prompt_no_styles = shared.prompt_styles.extract_styles_from_prompt(hr_prompt, hr_negative_prompt)
            if same_hr_styles := found_styles == hr_found_styles:
                res["Hires prompt"] = '' if hr_prompt_no_styles == prompt_no_styles else hr_prompt_no_styles
                res['Hires negative prompt'] = '' if hr_negative_prompt_no_styles == negative_prompt_no_styles else hr_negative_prompt_no_styles

        if same_hr_styles:
            prompt, negative_prompt = prompt_no_styles, negative_prompt_no_styles
            if (shared.opts.infotext_styles == "Apply if any" and found_styles) or shared.opts.infotext_styles == "Apply":
                res['Styles array'] = found_styles

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires schedule type" not in res:
        res["Hires schedule type"] = "Use same scheduler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    if "Mask mode" not in res:
        res["Mask mode"] = "Inpaint masked"

    if "Masked content" not in res:
        res["Masked content"] = 'original'

    if "Inpaint area" not in res:
        res["Inpaint area"] = "Whole picture"

    if "Masked area padding" not in res:
        res["Masked area padding"] = 32

    restore_old_hires_fix_params(res)

    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Sigma max" not in res:
        res["Sigma max"] = 0

    if "Sigma min" not in res:
        res["Sigma min"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    if "VAE Encoder" not in res:
        res["VAE Encoder"] = "Full"

    if "VAE Decoder" not in res:
        res["VAE Decoder"] = "Full"

    prompt_attention = prompt_parser.parse_prompt_attention(prompt)
    prompt_attention += prompt_parser.parse_prompt_attention(negative_prompt)
    prompt_uses_emphasis = len(prompt_attention) != len([p for p in prompt_attention if p[1] == 1.0 or p[0] == 'BREAK'])
    if "Emphasis" not in res and prompt_uses_emphasis:
        res["Emphasis"] = "Original"

    if "Tiling" in res:
        if res["Tiling"] == "True":
            res["Tiling"] = "X and Y"

    infotext_versions.backcompat(res)

    for key in skip_fields:
        res.pop(key, None)

    # basic check for same checkpoint using short name
    checkpoint = res.get('Model', None)
    if checkpoint is not None:
        if checkpoint in shared.opts.sd_model_checkpoint:
            res.pop('Model')

    # VAE / TE
    modules = []
    hr_modules = []
    vae = res.pop('VAE', None)  # old form
    if vae:
        modules = [vae]
    else:
        for key in res:
            if key.startswith('Module '):
                added = False
                for knownmodule in main_entry.module_list.keys():
                    filename, _ = os.path.splitext(knownmodule)
                    if res[key] == filename:
                        added = True
                        modules.append(knownmodule)
                        break
                if not added:
                    modules.append(res[key])   # so it shows in the override section (consistent with checkpoint and old vae)
            elif key.startswith('Hires Module '):
                for knownmodule in main_entry.module_list.keys():
                    filename, _ = os.path.splitext(knownmodule)
                    if res[key] == filename:
                        hr_modules.append(knownmodule)
                        break

    if modules != []:
        current_modules = shared.opts.forge_additional_modules
        basename_modules = []
        for m in current_modules:
            basename_modules.append(os.path.basename(m))

        if sorted(modules) != sorted(basename_modules):
            res['VAE/TE'] = modules

    # if 'Use same choices' was the selection for Hires VAE / Text Encoder, it will be the only Hires Module
    # if the selection was empty, it will be the only Hires Module, saved as 'Built-in'
    if 'Hires Module 1' in res:
        if res['Hires Module 1'] == 'Use same choices':
            hr_modules = ['Use same choices']
        elif res['Hires Module 1'] == 'Built-in':
            hr_modules = []

        res['Hires VAE/TE'] = hr_modules
    else:
        # no Hires Module infotext, use default
        res['Hires VAE/TE'] = ['Use same choices']

    return res


infotext_to_setting_name_mapping = [
    ('VAE/TE', 'forge_additional_modules'),
]
"""Mapping of infotext labels to setting names. Only left for backwards compatibility - use OptionInfo(..., infotext='...') instead.
Example content:

infotext_to_setting_name_mapping = [
    ('Conditional mask weight', 'inpainting_mask_weight'),
    ('Model hash', 'sd_model_checkpoint'),
    ('ENSD', 'eta_noise_seed_delta'),
    ('Schedule type', 'k_sched_type'),
]
"""
from ast import literal_eval
def create_override_settings_dict(text_pairs):
    """creates processing's override_settings parameters from gradio's multiselect

    Example input:
        ['Clip skip: 2', 'Model hash: e6e99610c4', 'ENSD: 31337']

    Example output:
        {'CLIP_stop_at_last_layers': 2, 'sd_model_checkpoint': 'e6e99610c4', 'eta_noise_seed_delta': 31337}
    """

    res = {}

    if not text_pairs:
        return res

    params = {}
    for pair in text_pairs:
        k, v = pair.split(":", maxsplit=1)

        params[k] = v.strip()

    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
    for param_name, setting_name in mapping + infotext_to_setting_name_mapping:
        value = params.get(param_name, None)

        if value is None:
            continue

        if setting_name == "forge_additional_modules":
            res[setting_name] = literal_eval(value)
            continue

        res[setting_name] = shared.opts.cast_value(setting_name, value)

    return res


def get_override_settings(params, *, skip_fields=None):
    """Returns a list of settings overrides from the infotext parameters dictionary.

    This function checks the `params` dictionary for any keys that correspond to settings in `shared.opts` and returns
    a list of tuples containing the parameter name, setting name, and new value cast to correct type.

    It checks for conditions before adding an override:
    - ignores settings that match the current value
    - ignores parameter keys present in skip_fields argument.

    Example input:
        {"Clip skip": "2"}

    Example output:
        [("Clip skip", "CLIP_stop_at_last_layers", 2)]
    """

    res = []

    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
    for param_name, setting_name in mapping + infotext_to_setting_name_mapping:
        if param_name in (skip_fields or {}):
            continue

        v = params.get(param_name, None)
        if v is None:
            continue

        if setting_name in ["sd_model_checkpoint", "forge_additional_modules"]:
            if shared.opts.disable_weights_auto_swap:
                continue

        v = shared.opts.cast_value(setting_name, v)
        current_value = getattr(shared.opts, setting_name, None)

        if v == current_value:
            continue

        res.append((param_name, setting_name, v))

    return res


def connect_paste(button, paste_fields, input_comp, override_settings_component, tabname):
    def paste_func(prompt):
        if not prompt and not shared.cmd_opts.hide_ui_dir_config and not shared.cmd_opts.no_prompt_history:
            filename = os.path.join(data_path, "params.txt")
            try:
                with open(filename, "r", encoding="utf8") as file:
                    prompt = file.read()
            except OSError:
                pass

        params = parse_generation_parameters(prompt)
        script_callbacks.infotext_pasted_callback(prompt, params)
        res = []

        for output, key in paste_fields:
            if callable(key):
                try:
                    v = key(params)
                except Exception:
                    errors.report(f"Error executing {key}", exc_info=True)
                    v = None
            else:
                v = params.get(key, None)

            if v is None:
                res.append(gr.update())
            elif isinstance(v, type_of_gr_update):
                res.append(v)
            else:
                try:
                    valtype = type(output.value)

                    if valtype == bool and v == "False":
                        val = False
                    elif valtype == int:
                        val = float(v)
                    else:
                        val = valtype(v)

                    res.append(gr.update(value=val))
                except Exception:
                    res.append(gr.update())

        return res

    if override_settings_component is not None:
        already_handled_fields = {key: 1 for _, key in paste_fields}

        def paste_settings(params):
            vals = get_override_settings(params, skip_fields=already_handled_fields)

            vals_pairs = [f"{infotext_text}: {value}" for infotext_text, setting_name, value in vals]

            return gr.Dropdown.update(value=vals_pairs, choices=vals_pairs, visible=bool(vals_pairs))

        paste_fields = paste_fields + [(override_settings_component, paste_settings)]

    button.click(
        fn=paste_func,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
        show_progress=False,
    )
    # button.click(
        # fn=None,
        # _js=f"recalculate_prompts_{tabname}",
        # inputs=None,
        # outputs=None,
        # show_progress=False,
    # )


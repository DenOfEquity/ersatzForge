import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared, ui_common, sd_models, processing, infotext_utils, paths, ui_loadsave
from backend import memory_management, stream
from backend.args import dynamic_args

total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Dropdown = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Number = None

ui_forge_unet_storage_dtype_options: gr.Dropdown = None
ui_forge_inference_memory: gr.Number = None
ui_forge_swap: gr.Dropdown = None


forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'float16 (fp16 LoRA)': (torch.float16, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

ckpt_list = []
module_list = {}        # vae + te + other
module_vae_list = {}
module_te_list = {}

def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], show_progress="hidden")
    return


def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
                # found_files[os.path.splitext(file)[0]] = full_path
    return found_files


def refresh_ckpt():
    global ckpt_list
    sd_models.list_models()
    ckpt_list = sd_models.checkpoint_tiles()
    
    
def refresh_vaete():
    global module_list, module_vae_list, module_te_list
    
    file_extensions = ['ckpt', 'pt', 'bin', 'safetensors', 'sft', 'gguf']

    module_list.clear()
    module_vae_list.clear()
    module_te_list.clear()

    # VAE
    vae_files = find_files_with_extensions(os.path.abspath(os.path.join(paths.models_path, "VAE")), file_extensions)
    module_vae_list.update(vae_files)
    if isinstance(shared.cmd_opts.vae_dir, str):
        vae_files = find_files_with_extensions(os.path.abspath(shared.cmd_opts.vae_dir), file_extensions)
        module_vae_list.update(vae_files)
    module_list.update(module_vae_list)

    # TE
    te_files = find_files_with_extensions(os.path.abspath(os.path.join(paths.models_path, "text_encoder")), file_extensions)
    module_te_list.update(te_files)
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        te_files = find_files_with_extensions(os.path.abspath(shared.cmd_opts.text_encoder_dir), file_extensions)
        module_te_list.update(te_files)
    module_list.update(module_te_list)

    # other
    other_list = find_files_with_extensions(os.path.abspath(os.path.join(paths.models_path, "other_module")), file_extensions)
    module_list.update(other_list)

    return module_list.keys()


def refresh_models():
    global ckpt_list
    refresh_ckpt()
    modules_list = refresh_vaete()

    return ckpt_list, modules_list

refresh_ckpt()
_ = refresh_vaete()

def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_swap, ui_forge_inference_memory, ui_forge_preset, ckpt_list

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Dropdown(label="UI", elem_id="forge_ui_preset", value='-', 
                                  choices=['sd', 'xl', 'sd3', 'flux', 'chroma', 'zimage', 'all'], scale=0, filterable=False)
    ui_checkpoint = gr.Dropdown(
        value=lambda: shared.opts.sd_model_checkpoint,
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list
    )

    ui_vae = gr.Dropdown(
        value=lambda: [os.path.basename(x) for x in shared.opts.forge_additional_modules],
        multiselect=True,
        label="Additional modules",
        elem_classes=['module_selection'],
        choices=list(module_list.keys())
    )

    def gr_refresh_models():    # updates HiRes fix checkpoint/modules too
        a, b = refresh_models()
        return gr.update(choices=a), gr.update(choices=b), gr.update(choices=["Use same checkpoint"] + a), gr.update(choices=["Use same choices"] + list(b))

    ui_txt2img_hr_checkpoint = get_a1111_ui_component('txt2img', 'HiRes checkpoint')
    ui_txt2img_hr_vae = get_a1111_ui_component('txt2img', 'HiRes additional modules')

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh_models,
        inputs=None,
        outputs=[ui_checkpoint, ui_vae, ui_txt2img_hr_checkpoint, ui_txt2img_hr_vae],
        show_progress="hidden",
    )

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()), filterable=False)
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)


    ui_forge_swap = gr.Dropdown(label="Swap Location + Method", value=lambda: shared.opts.forge_swap, choices=["CPU + Async", "CPU + Queue", "Shared + Async", "Shared + Queue"], filterable=False)
    ui_forge_inference_memory = gr.Number(label="Reserve VRAM (MB)", value=lambda: shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1, scale=0)

    mem_comps = [ui_forge_inference_memory, ui_forge_swap]

    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, show_progress="hidden")
    ui_forge_swap.change(ui_refresh_memory_management_settings, inputs=mem_comps, show_progress="hidden")

    Context.root_block.load(ui_refresh_memory_management_settings, inputs=mem_comps, show_progress="hidden")

    ui_clip_skip = gr.Number(label="Clip skip", value=lambda: shared.opts.CLIP_stop_at_last_layers, minimum=1, maximum=12, step=1, scale=0)
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    ui_checkpoint.change(checkpoint_change_ui, inputs=[ui_checkpoint, ui_vae], outputs=[ui_vae], show_progress="hidden")
    ui_vae.change(modules_change, inputs=[ui_vae], show_progress="hidden")

    return


def ui_refresh_memory_management_settings(inference_extra, swap):
    refresh_memory_management_settings(
        async_loading="Async" if "Async" in swap else "Queue",
        pin_shared_memory="CPU" if "CPU" in swap else "Shared",
        inference_memory=inference_extra
    )

def refresh_memory_management_settings(async_loading="Queue", pin_shared_memory="CPU", inference_memory=None):
    # Fallback to defaults if values are not passed
    if inference_memory is None:
        inference_memory = shared.opts.forge_inference_memory

    shared.opts.set('forge_swap', f'{pin_shared_memory} + {async_loading}')
    shared.opts.set('forge_inference_memory', inference_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.extra_inference_memory = inference_memory * 1024 * 1024  # Convert MB to bytes
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    checkpoint_info = sd_models.select_checkpoint()

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    dynamic_args['online_lora'] = lora_fp16

    sd_models.model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    print(f'Model selected: {sd_models.model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')

    return


def checkpoint_change_ui(ckpt_name:str, vae_te:list):
    result = vae_te

    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info != current_ckpt_info:
        if shared.opts.sd_vae_overrides_per_model_preferences:
            from modules import extra_networks

            metadata = extra_networks.get_user_metadata(new_ckpt_info.filename)
            vae_metadata = metadata.get("vae_te", None)
            if vae_metadata is None:
                vae_metadata = metadata.get("vae", None)

            if vae_metadata is not None:
                if isinstance(vae_metadata, str):
                    vae_metadata = [vae_metadata]

                if "Built in" in vae_metadata:  # this means use models built in to checkpoint, so clear the selection
                    vae_metadata = []

                if vae_metadata != ['']:        # ['']  means 'no change', keep whatever is already set
                    modules_change(vae_metadata, save=False, refresh=False)
                    result = vae_metadata

        shared.opts.set('sd_model_checkpoint', ckpt_name)

        shared.opts.save(shared.config_filename)
        refresh_model_loading_parameters()

    return result


def checkpoint_change(ckpt_name:str, save=True, refresh=True):
    """ checkpoint name can be a number of valid aliases. Returns True if checkpoint changed. """
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set('sd_model_checkpoint', ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def modules_change(module_values:list, save=True, refresh=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])

    # skip further processing if value unchanged
    if sorted(modules) == sorted(shared.opts.data.get('forge_additional_modules', [])):
        return False

    shared.opts.set('forge_additional_modules', modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    ui_txt2img_hr_cfg = get_a1111_ui_component('txt2img', 'HiRes CFG scale')
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'HiRes Distilled CFG scale')

    ui_txt2img_steps = get_a1111_ui_component('txt2img', 'steps')
    ui_img2img_steps = get_a1111_ui_component('img2img', 'steps')

    output_targets = [
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_inference_memory,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_hr_cfg,
        ui_txt2img_hr_distilled_cfg,
        ui_txt2img_steps,
        ui_img2img_steps,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, show_progress="hidden").then(
                          js="clickLoraRefresh", fn=None, show_progress="hidden")

# not setting on startup
#    Context.root_block.load(on_preset_change, inputs=None, outputs=output_targets, show_progress="hidden")

    refresh_model_loading_parameters()
    return


def on_preset_change(preset=None):
    if preset is None:
        preset = getattr(shared.opts, "forge_preset", "all")
    else:
        shared.opts.set("forge_preset", preset)
        shared.opts.save(shared.config_filename)

    if preset == "all":
        if shared.opts.use_ui_config_json:
            loadsave = ui_loadsave.UiLoadsave(shared.cmd_opts.ui_config_file)
            ui_settings_from_file = loadsave.ui_settings.copy()

            return [
                gr.skip(),
                gr.update(visible=True),
                gr.skip(),
                gr.update(value=0),
                gr.update(value=ui_settings_from_file['txt2img/Width/value']),
                gr.update(value=ui_settings_from_file['img2img/Width/value']),
                gr.update(value=ui_settings_from_file['txt2img/Height/value']),
                gr.update(value=ui_settings_from_file['img2img/Height/value']),
                gr.update(value=ui_settings_from_file['txt2img/CFG scale/value']),
                gr.update(value=ui_settings_from_file['img2img/CFG scale/value']),
                gr.update(visible=True, value=ui_settings_from_file['txt2img/Distilled CFG scale/value']),
                gr.update(visible=True, value=ui_settings_from_file['img2img/Distilled CFG scale/value']),
                gr.update(value=ui_settings_from_file['customscript/sampler.py/txt2img/Sampling method/value']),
                gr.update(value=ui_settings_from_file['customscript/sampler.py/img2img/Sampling method/value']),
                gr.update(value=ui_settings_from_file['customscript/sampler.py/txt2img/Schedule type/value']),
                gr.update(value=ui_settings_from_file['customscript/sampler.py/img2img/Schedule type/value']),
                gr.update(visible=True, value=ui_settings_from_file['txt2img/HiRes CFG scale/value']),
                gr.update(visible=True, value=ui_settings_from_file['txt2img/HiRes Distilled CFG scale/value']),
                gr.skip(),
                gr.skip(),
            ]
        else:
            return [
                gr.skip(),
                gr.update(visible=True),
                gr.skip(),
                gr.update(value=0),
                gr.update(value=1024),
                gr.update(value=1024),
                gr.update(value=1024),
                gr.update(value=1024),
                gr.update(value=5),
                gr.update(value=5),
                gr.update(visible=True, value=3.5),
                gr.update(visible=True, value=3.5),
                gr.update(value="Euler"),
                gr.update(value="Euler"),
                gr.update(value="Simple"),
                gr.update(value="Simple"),
                gr.update(value=3),
                gr.update(visible=True, value=3.5),
                gr.skip(),
                gr.skip(),
            ]
    else: # other presets
        preset_code = getattr(shared.opts, f"preset_code_{preset}", "")
        codes = preset_code.split(",")
        if len(codes) == 9:
            codes = [c.strip() for c in codes]
            return [
                gr.skip(),          #vae_te
                gr.update(visible=False) if codes[0] == "None" else gr.update(visible=True, value=int(codes[0])),   # ui_clip_skip
                gr.skip(),  #storage type
                gr.update(visible=False, value=0) if codes[1] == "None" else gr.update(visible=True, value=int(codes[1])),   # mem
                gr.update(value=int(codes[2])),
                gr.update(value=int(codes[2])),
                gr.update(value=int(codes[3])),
                gr.update(value=int(codes[3])),
                gr.update(visible=False, value=1.0) if codes[4] == "None" else gr.update(visible=True, value=float(codes[4])), #cfg t2i
                gr.update(visible=False, value=1.0) if codes[4] == "None" else gr.update(visible=True, value=float(codes[4])), #cfg i2i
                gr.update(visible=False, value=0.0) if codes[5] == "None" else gr.update(visible=True, value=float(codes[5])), #dcfg t2i
                gr.update(visible=False, value=0.0) if codes[5] == "None" else gr.update(visible=True, value=float(codes[5])), #dcfg i2i
                gr.update(value=codes[6]), #sampler
                gr.update(value=codes[6]),
                gr.update(value=codes[7]), #scheduler
                gr.update(value=codes[7]),
                gr.update(visible=False, value=1.0) if codes[4] == "None" else gr.update(visible=True, value=float(codes[4])), #cfg hr
                gr.update(visible=False, value=0.0) if codes[5] == "None" else gr.update(visible=True, value=float(codes[5])), #dcfg hr
                gr.update(value=int(codes[8])), #steps
                gr.update(value=int(codes[8])),
            ]
            
        else:
            model_mem = getattr(shared.opts, f"{preset}_GPU_MB", 0)
            if model_mem < 0 or model_mem > total_vram:
                model_mem = 0

            p = "flux" if preset == "chroma" else preset
            return [
                gr.update(value=getattr(shared.opts, f"{p}_vae_te", [""])),
                gr.update(visible=(p != "flux")),                                                          # ui_clip_skip
                gr.update(value=getattr(shared.opts, f"{p}_unet_dtype", "Automatic")),
                gr.update(value=model_mem),
                gr.update(value=getattr(shared.opts, f"{p}_t2i_width", 512)),
                gr.update(value=getattr(shared.opts, f"{p}_i2i_width", 512)),
                gr.update(value=getattr(shared.opts, f"{p}_t2i_height", 640)),
                gr.update(value=getattr(shared.opts, f"{p}_i2i_height", 512)),
                gr.update(value=getattr(shared.opts, f"{p}_t2i_cfg", 1)),
                gr.update(value=getattr(shared.opts, f"{p}_i2i_cfg", 1)),
                gr.update(visible=(preset == "flux"), value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)),       # ui_txt2img_distilled_cfg
                gr.update(visible=(preset == "flux"), value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)),       # ui_img2img_distilled_cfg
                gr.update(value=getattr(shared.opts, f"{p}_t2i_sampler", "Euler")),
                gr.update(value=getattr(shared.opts, f"{p}_i2i_sampler", "Euler")),
                gr.update(value=getattr(shared.opts, f"{p}_t2i_scheduler", "Simple")),
                gr.update(value=getattr(shared.opts, f"{p}_i2i_scheduler", "Simple")),
                gr.update(value=getattr(shared.opts, f"{p}_t2i_hr_cfg", 1.0)),
                gr.update(visible=(preset == "flux"), value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),    # ui_txt2img_hr_distilled_cfg
                gr.skip(),
                gr.skip(),
            ]

shared.options_templates.update(shared.options_section(('ui_other', "UI defaults (other)", "ui"), {
    "preset_codes_explanation": shared.OptionHTML("""
<h3>Codes for UI presets.</h3>
All settings apply to <strong>Txt2img</strong> and <strong>Img2img</strong>.<br/>
Code order is: (all must be present)<br/>
<ol>
<li>clip skip : <em>can be </em>None<em>, in which case the control will be hidden</em> <sub>note: value can be changed by loading Infotext</sub></li>
<li>reserved VRAM : <em>can be </em>None</li>
<li>width</li>
<li>height</li>
<li>CFG : <em>also applies to <strong>HiRes-fix</strong>; can be </em>None</li>
<li>distilled CFG : <em>also applies to <strong>HiRes-fix</strong>; can be </em>None</li>
<li>sampler</li>
<li>scheduler</li>
<li>sampling steps</li>
</ol>
"""),
    "preset_code_chroma": shared.OptionInfo("", "Chroma preset code", gr.Textbox, {"maxlines": 1, "placeholder": "see UI defaults 'flux'"}),
    "preset_code_zimage": shared.OptionInfo("None, 0, 1024, 1024, 1.0, None, Euler, Simple, 8", "Zimage preset code", gr.Textbox, {"maxlines": 1}),

}))

shared.options_templates.update(shared.options_section(('ui_sd', "UI defaults 'sd'", "ui"), {
    "sd_t2i_width":  shared.OptionInfo(512,  "txt2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_t2i_height": shared.OptionInfo(640,  "txt2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_t2i_cfg":    shared.OptionInfo(7,    "txt2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd_t2i_hr_cfg": shared.OptionInfo(7,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd_i2i_width":  shared.OptionInfo(512,  "img2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_i2i_height": shared.OptionInfo(512,  "img2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_i2i_cfg":    shared.OptionInfo(7,    "img2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd_GPU_MB":     shared.OptionInfo(0,    "Reserve VRAM (MB)",  gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "sd_vae_te":     shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "sd_unet_dtype": shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))
shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults 'xl'", "ui"), {
    "xl_t2i_width":  shared.OptionInfo(896,  "txt2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_t2i_height": shared.OptionInfo(1152, "txt2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_t2i_cfg":    shared.OptionInfo(5,    "txt2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "xl_t2i_hr_cfg": shared.OptionInfo(5,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "xl_i2i_width":  shared.OptionInfo(1024, "img2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_i2i_height": shared.OptionInfo(1024, "img2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_i2i_cfg":    shared.OptionInfo(5,    "img2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "xl_GPU_MB":     shared.OptionInfo(0,    "Reserve VRAM (MB)",  gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "xl_vae_te":     shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "xl_unet_dtype": shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))
shared.options_templates.update(shared.options_section(('ui_sd3', "UI defaults 'sd3'", "ui"), {
    "sd3_t2i_width":  shared.OptionInfo(896,  "txt2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_t2i_height": shared.OptionInfo(1152, "txt2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_t2i_cfg":    shared.OptionInfo(5,    "txt2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd3_t2i_hr_cfg": shared.OptionInfo(5,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd3_i2i_width":  shared.OptionInfo(1024, "img2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_i2i_height": shared.OptionInfo(1024, "img2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_i2i_cfg":    shared.OptionInfo(5,    "img2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd3_GPU_MB":     shared.OptionInfo(0,    "Reserve VRAM (MB)",  gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "sd3_vae_te":     shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "sd3_unet_dtype": shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))
shared.options_templates.update(shared.options_section(('ui_flux', "UI defaults 'flux'", "ui"), {
    "flux_t2i_width":    shared.OptionInfo(896,  "txt2img width",                gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_t2i_height":   shared.OptionInfo(1152, "txt2img height",               gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_t2i_cfg":      shared.OptionInfo(1,    "txt2img CFG",                  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_cfg":   shared.OptionInfo(1,    "txt2img HiRes CFG",            gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "flux_t2i_d_cfg":    shared.OptionInfo(3.5,  "txt2img Distilled CFG",        gr.Slider, {"minimum": 0,   "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_d_cfg": shared.OptionInfo(3.5,  "txt2img Distilled HiRes CFG",  gr.Slider, {"minimum": 0,   "maximum": 30,   "step": 0.1}),
    "flux_i2i_width":    shared.OptionInfo(1024, "img2img width",                gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_i2i_height":   shared.OptionInfo(1024, "img2img height",               gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_i2i_cfg":      shared.OptionInfo(1,    "img2img CFG",                  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "flux_i2i_d_cfg":    shared.OptionInfo(3.5,  "img2img Distilled CFG",        gr.Slider, {"minimum": 0,   "maximum": 30,   "step": 0.1}),
    "flux_GPU_MB":       shared.OptionInfo(0,    "Reserve VRAM (MB)",            gr.Slider, {"minimum": 0,   "maximum": total_vram,   "step": 1}),
    "flux_vae_te":       shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "flux_unet_dtype":   shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))


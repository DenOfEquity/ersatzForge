import os
import gradio as gr

from modules import localization, ui_components, shared_items, shared, shared_gradio_themes, util
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, default_output_dir  # noqa: F401
from modules.shared_cmd_options import cmd_opts
from modules.options import options_section, OptionInfo, OptionHTML, categories
from modules_forge import shared_options as forge_shared_options

from backend.text_processing import emphasis

options_templates = {}
hide_dirs = shared.hide_dirs

restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "temp_dir",
    "clean_temp_dir_at_start",
}

categories.register_category("saving", "Saving images")
categories.register_category("sd", "Stable Diffusion")
categories.register_category("ui", "User Interface")
categories.register_category("system", "System")
categories.register_category("postprocessing", "Postprocessing")

options_templates.update(options_section(('saving-images', "Saving images/grids", "saving"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "save_images_replace_action": OptionInfo("Replace", "Saving the image to an existing file", gr.Radio, {"choices": ["Replace", "Add number suffix"], **hide_dirs}),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('png', 'File format for grids'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "font": OptionInfo("", "Font for image grids that have text"),
    "grid_text_active_color": OptionInfo("#000000", "Text color for image grids", ui_components.FormColorPicker, {}),
    "grid_text_inactive_color": OptionInfo("#999999", "Inactive text color for image grids", ui_components.FormColorPicker, {}),
    "grid_background_color": OptionInfo("#ffffff", "Background color for image grids", ui_components.FormColorPicker, {}),

    "save_images_before_postprocess": OptionInfo(False, "Save a copy of image before running post-processing - face restoration, scripts, colour correction."),
    "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg and avif images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "export_for_4chan": OptionInfo(True, "Save copy of large images as JPG").info("if the file size is above the limit, or either width or height are above the limit"),
    "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option, MB", gr.Number),
    "target_side_length": OptionInfo(4000, "Width/height limit for the above option, in pixels", gr.Number),
    "img_max_size_mp": OptionInfo(200, "Maximum image size", gr.Number).info("in megapixels"),

    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),

    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),
}))

options_templates.update(options_section(('saving-paths', "Paths for saving", "saving"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'txt2img-images')), 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'img2img-images')), 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'extras-images')), 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'txt2img-grids')), 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, 'img2img-grids')), 'Output directory for img2img grids', component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory", "saving"), {
    "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('upscaling', "Upscaling", "postprocessing"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}).info("Low values = visible seam"),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI.", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "DAT_tile": OptionInfo(192, "Tile size for DAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
    "DAT_tile_overlap": OptionInfo(8, "Tile overlap for DAT upscalers.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}).info("Low values = visible seam"),

    "SWIN_tile": OptionInfo(192, "Tile size for SwinIR upscalers.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}),
    "SWIN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "SWIN_torch_compile": OptionInfo(False, "Use torch.compile to accelerate SwinIR.").info("Takes longer on first run"),
#can these tile size/overlap be shared?
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in shared.sd_upscalers]}),
}))

options_templates.update(options_section(('face-restoration', "Face restoration", "postprocessing"), {
    "face_restoration_model": OptionInfo("None", "Face restoration model", gr.Dropdown, lambda: {"choices": ["None"] + [x.name() for x in shared.face_restorers], "filterable": False}),
    "face_restoration_before_scripts": OptionInfo(True, "Apply face restoration before running postprocess scripts. Disabled = apply after scripts."),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}).info("0 = maximum effect; 1 = minimum effect"),
}))

options_templates.update(options_section(('system', "System", "system"), {
    "auto_launch_browser": OptionInfo("Local", "Automatically open webui in browser on startup", gr.Radio, lambda: {"choices": ["Disable", "Local", "Remote"]}),
    "show_warnings": OptionInfo(False, "Show warnings in console.").needs_reload_ui(),
    "show_gradio_deprecation_warnings": OptionInfo(True, "Show gradio deprecation warnings in console.").needs_reload_ui(),
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}).info("0 = disable"),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
    "print_hypernet_extra": OptionInfo(False, "Print extra hypernetwork information to console."),
    "list_hidden_files": OptionInfo(True, "Load models/files in hidden directories").info("directory is hidden if its name starts with \".\""),
    "dump_stacks_on_signal": OptionInfo(False, "Print stack traces before exiting the program with ctrl+c."),
    "use_real_SHA256": OptionInfo(False, "Use SHA256 to hash checkpoints, done on first use only. Warning: this can take a significant amount of time on large checkpoints."),
}))

options_templates.update(options_section(('profiler', "Profiler", "system"), {
    "profiling_explanation": OptionHTML("""
Those settings allow you to enable torch profiler when generating pictures.
Profiling allows you to see which code uses how much of computer's resources during generation.
Each generation writes its own profile to one file, overwriting previous.
The file can be viewed in <a href="chrome:tracing">Chrome</a>, or on a <a href="https://ui.perfetto.dev/">Perfetto</a> web site.
Warning: writing profile can take a lot of time, up to 30 seconds, and the file itelf can be around 500MB in size.
"""),
    "profiling_enable": OptionInfo(False, "Enable profiling"),
    "profiling_activities": OptionInfo(["CPU"], "Activities", gr.CheckboxGroup, {"choices": ["CPU", "CUDA"]}),
    "profiling_record_shapes": OptionInfo(True, "Record shapes"),
    "profiling_profile_memory": OptionInfo(True, "Profile memory"),
    "profiling_with_stack": OptionInfo(True, "Include python stack"),
    "profiling_filename": OptionInfo("trace.json", "Profile filename"),
}))

options_templates.update(options_section(('API', "API", "system"), {
    "api_enable_requests": OptionInfo(True, "Allow http:// and https:// URLs for input images in API", restrict_api=True),
    "api_forbid_local_requests": OptionInfo(True, "Forbid URLs to local resources", restrict_api=True),
    "api_useragent": OptionInfo("", "User agent for requests", restrict_api=True),
}))

options_templates.update(options_section(('sd', "Stable Diffusion", "sd"), {
    "sd_model_checkpoint": OptionInfo(None, "(Managed by Forge)", gr.State, infotext="Model"),
    "emphasis": OptionInfo("Original", "Emphasis mode", gr.Radio, lambda: {"choices": [x.name for x in emphasis.options]}, infotext="Emphasis").info("makes it possible to make model to pay (more:1.1) or (less:0.9) attention to text when you use the syntax in prompt; " + emphasis.get_options_descriptions()),
    "comma_padding_backtrack": OptionInfo(20, "Prompt word wrap length limit", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("in tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"),
    "CLIP_stop_at_last_layers": OptionInfo(1, "(Managed by Forge)", gr.State, infotext="Clip skip"),
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
    "randn_source": OptionInfo("GPU", "Random number generator source.", gr.Radio, {"choices": ["GPU", "CPU", "NV", "Perlin"]}, infotext="RNG").info("changes seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"),
    "perlin_detail": OptionInfo(1.0, "Perlin noise detail level", gr.Slider, {"minimum": 0, "maximum": 2, "step": 0.01}, infotext="Perlin detail"),
    "perlin_octaves": OptionInfo(1, "Perlin noise octaves", gr.Slider, {"minimum": 1, "maximum": 7, "step": 1}, infotext="Perlin octaves").info("2**(octaves-1) must divide exactly into latent width and height. Value will be clamped automatically when used."),
    "perlin_persist": OptionInfo(1.0, "Perlin noise persistence", gr.Slider, {"minimum": 0.01, "maximum": 2, "step": 0.01}, infotext="Perlin persistence").info("Persistence is the influence of successive octaves of noise."),
    "tiling": OptionInfo("None", "Tiling", gr.Radio, {"choices": ["None", "X", "Y", "X and Y"]}, infotext='Tiling').info("produce a tileable picture"),
    "hires_fix_refiner_pass": OptionInfo("second pass", "Hires fix: which pass to enable refiner for", gr.Radio, {"choices": ["first pass", "second pass", "both passes"]}, infotext="Hires refiner"),
    "use_ELLA": OptionInfo("CLIP (normal)", "Use ELLA for SD1.5", gr.Radio, {"choices": ["CLIP (normal)", "ELLA only", "ELLA (per step) only", "CLIP + ELLA", "CLIP + ELLA (per step)"]}, infotext="ELLA").info("ELLA text encoder and ELLA model will be automatically downloaded. Info: https://github.com/TencentQQGYLab/ELLA"),
    "epsilon_scaling": OptionInfo(1.0, "Epsilon scaling factor", gr.Slider, {"minimum": 0.8, "maximum": 1.2, "step": 0.001}, infotext="Epsilon scaling").info("'Elucidating the Exposure Bias in Diffusion Models' https://openreview.net/pdf?id=xEJMoj1SpX"),
    "epsilon_modulation": OptionInfo(False, "Epsilon modulation", infotext="Epsilon modulation").info("reduces effect of Epsilon scaling on later steps (sigma / sigma_max)"),
    "prediction_scaling": OptionInfo(1.0, "Prediction scaling factor", gr.Slider, {"minimum": 0.8, "maximum": 1.2, "step": 0.001}, infotext="Prediction scaling").info("Scales model output, with modulation by sigma"),
}))

options_templates.update(options_section(('sdxl', "Stable Diffusion XL", "sd"), {
    "sdxl_crop_top": OptionInfo(0, "crop top coordinate", gr.Number, {"minimum": 0, "maximum": 1024, "step": 1}),
    "sdxl_crop_left": OptionInfo(0, "crop left coordinate", gr.Number, {"minimum": 0, "maximum": 1024, "step": 1}),
    "sdxl_refiner_low_aesthetic_score": OptionInfo(2.5, "SDXL low aesthetic score", gr.Slider, {"minimum": 0, "maximum": 10, "step": 0.1}).info("used for refiner model negative prompt"),
    "sdxl_refiner_high_aesthetic_score": OptionInfo(6.0, "SDXL high aesthetic score", gr.Slider, {"minimum": 0, "maximum": 10, "step": 0.1}).info("used for refiner model prompt"),
    "sdxl_flow_shift": OptionInfo(3.0, "Flow Shift for SDXL flow match models. Relevant models are detected by name.", gr.Slider, {"minimum": 0.01, "maximum": 12.0, "step": 0.01}, infotext="SDXL Shift"),
}))

options_templates.update(options_section(('vae', "VAE", "sd"), {
    "sd_vae_explanation": OptionHTML("""
<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>
image into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling
(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.
For img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.
"""),
    "sd_vae": OptionInfo("Automatic", "(Managed by Forge)", gr.State, infotext='VAE'),
    "sd_vae_overrides_per_model_preferences": OptionInfo(True, "Selected VAE / Text Encoder per-model preferences").info("you can set per-model VAE / Text Encoder by editing user metadata for checkpoints"),
    "sd_vae_encode_method": OptionInfo("Full", "VAE type for encode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Encoder').info("method to encode image to latent (use in img2img, hires-fix or inpaint mask)"),
    "sd_vae_decode_method": OptionInfo("Full", "VAE type for decode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Decoder').info("method to decode latent to image"),
}))

options_templates.update(options_section(('img2img', "img2img", "sd"), {
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Conditional mask weight'),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.5, "step": 0.001}, infotext='Noise multiplier'),
    "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img and hires fix", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Extra noise').info("0 = disabled (default); value will be clamped to <= 0.5 * denoising strength"),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill transparent parts of the input image with this color.", ui_components.FormColorPicker, {}),
    "img2img_inpaint_sketch_default_brush_color": OptionInfo("#ffffff", "Inpaint sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img inpaint sketch").needs_reload_ui(),
    "img2img_batch_show_results_limit": OptionInfo(32, "Show the first N batch img2img results in UI", gr.Slider, {"minimum": -1, "maximum": 1000, "step": 1}).info('0: disable, -1: show all images. Too many images can cause lag'),
    "overlay_inpaint": OptionInfo(True, "Overlay original for inpaint").info("when inpainting, overlay the original image over the areas that weren't inpainted."),
    "img2img_autosize": OptionInfo(False, "After loading into Img2img, automatically update Width and Height"),
    "img2img_batch_use_original_name": OptionInfo(False, "Save using original filename in img2img batch. Applies to 'Upload' and 'From directory' tabs.").info("Warning: overwriting is possible, based on Settings > Saving images/grids > Saving the image to an existing file."),

# compatibility
    "img2img_inpaint_mask_high_contrast": OptionInfo(True, "For inpainting, use a high-contrast brush pattern").info("use a checkerboard brush pattern instead of color brush").needs_reload_ui(),
    "img2img_inpaint_mask_brush_color": OptionInfo("#ffffff", "Inpaint mask brush color", ui_components.FormColorPicker,  {}).info("brush color of inpaint mask").needs_reload_ui(),
}))

options_templates.update(options_section(('optimizations', "Optimizations", "sd"), {
    "s_min_uncond": OptionInfo(0.0, "Negative Guidance minimum sigma", gr.Slider, {"minimum": 0.0, "maximum": 15.0, "step": 0.01}, infotext='NGMS').link("PR", "https://github.com/AUTOMATIC1111/stablediffusion-webui/pull/9177").info("skip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster"),
    "s_min_uncond_all": OptionInfo(False, "Negative Guidance minimum sigma all steps", infotext='NGMS all steps').info("By default, NGMS above skips every other step; this makes it skip all steps"),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256").info("0=disable, higher=faster"),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).info("only applies if non-zero and overrides above"),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for high-res pass", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio hr').info("only applies if non-zero and overrides above"),
    "persistent_cond_cache": OptionInfo(True, "Persistent cond cache").info("do not recalculate conds from prompts if prompts have not changed since previous calculation"),
}))

options_templates.update(options_section(('compatibility', "Compatibility", "sd"), {
    "forge_try_reproduce": OptionInfo('None', "Try to reproduce the results from external software", gr.Radio, lambda: {"choices": ['None', 'Diffusers', 'ComfyUI', 'WebUI 1.5', 'InvokeAI', 'EasyDiffusion', 'DrawThings']}),
    "auto_backcompat": OptionInfo(True, "Automatic backward compatibility").info("automatically enable options for backwards compatibility when importing generation parameters from infotext that has program version."),
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10).", infotext='Old Karras sigmas'),
    "hires_fix_use_firstpass_conds": OptionInfo(False, "For hires fix, calculate conds of second pass using extra networks of first pass."),
    "use_old_scheduling": OptionInfo(False, "Use old prompt editing timelines.", infotext="Old prompt editing timelines").info("For [red:green:N]; OLD: If N < 1, it's a fraction of steps (hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; NEW: If N has a decimal point in it, it's a fraction of steps (hires fix uses range from 1 to 2), otherwise it's an absolute number of steps"),
    "use_downcasted_alpha_bar": OptionInfo(False, "Downcast model alphas_cumprod to fp16 before sampling. For reproducing old seeds.", infotext="Downcast alphas_cumprod"),
}))

options_templates.update(options_section(('interrogate', "Interrogate"), {
    "interrogate_return_ranks": OptionInfo(False, "Include ranks of model tags matches in results.").info("booru only"),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "deepbooru: score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "deepbooru: sort tags alphabetically").info("if not: sort by score"),
    "deepbooru_use_spaces": OptionInfo(True, "deepbooru: use spaces in tags").info("if not: use underscores"),
    "deepbooru_escape": OptionInfo(True, "deepbooru: escape (\\) brackets").info("so they are used as literal brackets and not for emphasis"),
    "deepbooru_filter_tags": OptionInfo("", "deepbooru: filter out those tags").info("separate by comma"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks", "sd"), {
    "extra_networks_show_hidden_directories": OptionInfo(True, "Show hidden directories").info("directory is hidden if its name starts with \".\"."),
    "extra_networks_dir_button_function": OptionInfo(False, "Add a '/' to the beginning of directory buttons").info("Buttons will display the contents of the selected directory without acting as a search filter."),
    "extra_networks_hidden_models": OptionInfo("When searched", "Show cards for models in hidden directories", gr.Radio, {"choices": ["Always", "When searched", "Never"]}).info('"When searched" option will only show the item when the search string has 4 characters or more'),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks").info("in pixels"),
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks").info("in pixels"),
    "extra_networks_card_text_scale": OptionInfo(1.0, "Card text scale", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}).info("1 = original size"),
    "extra_networks_card_show_desc": OptionInfo(True, "Show description on card"),
    "extra_networks_card_description_is_html": OptionInfo(False, "Treat card description as HTML"),
    "extra_networks_card_order_field": OptionInfo("Path", "Default order field for Extra Networks cards", gr.Dropdown, {"choices": ['Path', 'Name', 'Date Modified']}).needs_reload_ui(),
    "extra_networks_card_order": OptionInfo("Ascending", "Default order for Extra Networks cards", gr.Dropdown, {"choices": ['Ascending', 'Descending']}).needs_reload_ui(),
    "extra_networks_tree_view_style": OptionInfo("Dirs", "Extra Networks directory view style", gr.Radio, {"choices": ["Tree", "Dirs"]}).needs_reload_ui(),
    "extra_networks_tree_view_default_enabled": OptionInfo(True, "Show the Extra Networks directory view by default").needs_reload_ui(),
    "extra_networks_tree_view_default_width": OptionInfo(180, "Default width for the Extra Networks directory tree view", gr.Number).needs_reload_ui(),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra networks separator").info("extra text to add before <...> when adding extra network to prompt"),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order").needs_reload_ui(),
    "textual_inversion_print_at_load": OptionInfo(False, "Print a list of Textual Inversion embeddings when loading model"),
}))

options_templates.update(options_section(('ui_prompt_editing', "Prompt editing", "ui"), {
    "keyedit_precision_attention": OptionInfo(0.1, "Precision for (attention:1.1) when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Precision for <extra networks:0.9> when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~() ", "Word delimiters when editing the prompt with Ctrl+up/down"),
    "keyedit_delimiters_whitespace": OptionInfo(["Tab", "Carriage Return", "Line Feed"], "Ctrl+up/down whitespace delimiters", gr.CheckboxGroup, lambda: {"choices": ["Tab", "Carriage Return", "Line Feed"]}),
    "disable_token_counters": OptionInfo(False, "Disable prompt token counters").needs_reload_ui(),
    "include_styles_into_token_counters": OptionInfo(True, "Count tokens of enabled styles").info("When calculating how many tokens the prompt has, also consider tokens added by enabled styles."),
}))

options_templates.update(options_section(('ui_gallery', "Gallery", "ui"), {
    "return_grid": OptionInfo(True, "Show grid in gallery"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in gallery"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Full page image viewer: show images zoomed in by default"),
    "gallery_height": OptionInfo("", "Gallery height", gr.Textbox).info("can be any valid CSS value, for example 768px or 20em").needs_reload_ui(),
    "open_dir_button_choice": OptionInfo("Subdirectory", "What directory the [📂] button opens", gr.Radio, {"choices": ["Output Root", "Subdirectory", "Subdirectory (even temp dir)"]}),
}))

options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
    "use_ui_config_json": OptionInfo(False, "Use 'ui-config.json' to store UI settings").needs_reload_ui(),
    "sd_checkpoint_dropdown_use_short": OptionInfo(False, "Checkpoint dropdown: use filenames without paths").info("models in subdirectories like photo/sd15.ckpt will be listed as just sd15.ckpt"),
    "txt2img_settings_accordion": OptionInfo(False, "Settings in txt2img hidden under Accordion").needs_reload_ui(),
    "img2img_settings_accordion": OptionInfo(False, "Settings in img2img hidden under Accordion").needs_reload_ui(),
}))

options_templates.update(options_section(('ui', "User interface", "ui"), {
    "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_reload_ui(),
    "quick_setting_list": OptionInfo([], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that appear at the top of page rather than in settings tab").needs_reload_ui(),
    "ui_tab_order": OptionInfo([], "UI tab order", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    "tabs_without_quick_settings_bar": OptionInfo(["Spaces"], "UI tabs without Quicksettings bar (top row)", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}),
    "ui_reorder_list": OptionInfo([], "UI item order for txt2img/img2img tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared_items.ui_reorder_categories())}).info("selected items appear first").needs_reload_ui(),
    "gradio_theme": OptionInfo("Default", "Gradio theme", ui_components.DropdownEditable, lambda: {"choices": ["Default"] + shared_gradio_themes.gradio_hf_hub_themes}).info("you can also manually enter any of themes from the <a href='https://huggingface.co/spaces/gradio/theme-gallery'>gallery</a>.").needs_reload_ui(),
    "gradio_themes_cache": OptionInfo(True, "Cache gradio themes locally").info("disable to update the selected Gradio theme"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "enable_reloading_ui_scripts": OptionInfo(False, "Reload UI scripts when using Reload UI option").info("useful for developing: if you make changes to UI scripts code, it is applied when the UI is reloded."),

    "hires_button_gallery_insert": OptionInfo(False, "Insert [✨] hires button results into gallery").info("Default: original image will be replaced"),
    "hires_button_iterate": OptionInfo("Disabled", "[✨] hires button iterates based on 'Batch count'", gr.Radio, {"choices": ["Disabled", "Enabled", "Enabled, Scaling"]}).info("Result is always one image. Enabled: uses same denoise and steps each iteration; Scaling reduces steps and denoise each iteration, ending at half of start values."),

}))


options_templates.update(options_section(('infotext', "Infotext", "ui"), {
    "infotext_explanation": OptionHTML("""
Infotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.
It is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button.
"""),
    "enable_pnginfo": OptionInfo(True, "Write infotext to metadata of the generated image"),
    "save_txt": OptionInfo(False, "Create a text file with infotext next to every generated image"),

    "add_model_name_to_info": OptionInfo(True, "Add model name to infotext"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to infotext"),
    "add_vae_name_to_info": OptionInfo(True, "Add VAE name to infotext"),
    "add_vae_hash_to_info": OptionInfo(True, "Add VAE hash to infotext"),
    "disable_weights_auto_swap": OptionInfo(True, "Disregard checkpoint information from pasted infotext").info("when reading generation parameters from text into UI"),
    "infotext_skip_pasting": OptionInfo([], "Disregard fields from pasted infotext", ui_components.DropdownMulti, lambda: {"choices": shared_items.get_infotext_names()}),
    "infotext_styles": OptionInfo("Apply if any", "Infer styles from prompts of pasted infotext", gr.Radio, {"choices": ["Ignore", "Apply", "Discard", "Apply if any"]}).info("when reading generation parameters from text into UI)").html("""<ul style='margin-left: 1.5em'>
<li>Ignore: keep prompt and styles dropdown as it is.</li>
<li>Apply: remove style text from prompt, always replace styles dropdown value with found styles (even if none are found).</li>
<li>Discard: remove style text from prompt, keep styles dropdown as it is.</li>
<li>Apply if any: remove style text from prompt; if any styles are found in prompt, put them into styles dropdown, otherwise keep it as it is.</li>
</ul>"""),

}))

options_templates.update(options_section(('ui', "Live previews", "ui"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "live_previews_image_format": OptionInfo("png", "Live preview file format", gr.Radio, {"choices": ["jpeg", "png", "webp"]}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "show_progress_every_n_steps": OptionInfo(10, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("in sampling steps - show new live preview image every N sampling steps; -1 = only show after completion of batch"),
    "show_progress_type": OptionInfo("Approx NN", "Live preview method", gr.Radio, {"choices": ["Approx NN", "Approx cheap", "TAESD"]}).info("Approx NN: fast preview; TAESD = high-quality preview; Approx cheap = fastest but low-quality preview"),
    "live_preview_refresh_period": OptionInfo(1000, "Progressbar and preview update period", gr.Number, {"minimum": 100, "maximum": 999999, "step": 1}).info("in milliseconds"),
    "js_live_preview_in_modal_lightbox": OptionInfo(False, "Show Live preview in full page image viewer"),
    "prevent_screen_sleep_during_generation": OptionInfo(True, "Prevent screen sleep during generation"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters", "sd"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in shared_items.list_samplers()]}).needs_reload_ui(),
    "hide_schedulers": OptionInfo([], "Hide schedulers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.label for x in shared_items.list_schedulers()]}).needs_reload_ui(),
    "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta DDIM').info("noise multiplier; higher = more unpredictable results"),
    "eta_ancestral": OptionInfo(1.0, "Eta for k-diffusion samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta').info("noise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 0.01}, infotext='Sigma churn').info('amount of stochasticity; only applies to Euler, Heun, and DPM2'),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.01}, infotext='Sigma tmin').info('enable stochasticity; start value of the sigma range; only applies to Euler, Heun, and DPM2'),
    's_tmax':  OptionInfo(0.0, "sigma tmax",  gr.Slider, {"minimum": 0.0, "maximum": 999.0, "step": 0.01}, infotext='Sigma tmax').info("0 = inf; end value of the sigma range; only applies to Euler, Heun, and DPM2"),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.1, "step": 0.001}, infotext='Sigma noise').info('amount of additional noise to counteract loss of detail during sampling'),
    'sigma_min': OptionInfo(0.0, "sigma min", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.001}, infotext='Sigma min').info("0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler"),
    'sigma_max': OptionInfo(0.0, "sigma max", gr.Slider, {"minimum": 0.0, "maximum": 60.0, "step": 0.001}, infotext='Sigma max').info("0 = default (~14.6); maximum noise strength for k-diffusion noise scheduler"),
    'rho':  OptionInfo(0.0, "rho", gr.Number, infotext='Schedule rho').info("0 = default (7 for karras, 1 for polyexponential); higher values result in a steeper noise schedule (decreases faster)"),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}, infotext='ENSD').info("ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma", infotext='Discard penultimate sigma').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"),
    'sgm_noise_multiplier': OptionInfo(False, "SGM noise multiplier", infotext='SGM noise multiplier').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818").info("Match initial noise to official SDXL implementation - only useful for reproducing images"),

    "adaptive_ode_explanation": OptionHTML("""
<h3>Adaptive-ODE relative and absolute tolerances need tuning to the solver</h3>
<ul>
<li><i>adaptive_heun</i> recommended: -2.5, -3.5</li>
<li><i>bosh3</i> recommended: -2.5, -3.5</li>
<li><i>fehlberg2</i> recommended: -4.0, -6.0</li>
</ul>
"""),
    'adaptive_ode_solver': OptionInfo("bosh3", "Adaptive-ODE solver", gr.Radio, {"choices": ["adaptive_heun", "bosh3", "fehlberg2"]}, infotext='Adaptive-ODE solver'),
    'adaptive_ode_rtol': OptionInfo(-2.5, "Adaptive-ODE log relative tolerance", gr.Slider, {"minimum": -7, "maximum": -1, "step": 0.01}, infotext='Adaptive-ODE rtol'),
    'adaptive_ode_atol': OptionInfo(-3.5, "Adaptive-ODE log absolute tolerance", gr.Slider, {"minimum": -7, "maximum": -1, "step": 0.01}, infotext='Adaptive-ODE atol'),

    'fixed_ode_solver': OptionInfo("rk4", "Fixed-ODE solver", gr.Radio, {"choices": ["implicit_adams", "heun3", "midpoint", "rk4"]}, infotext='Fixed-ODE solver'),

    'deis_mode': OptionInfo("tab", "DEIS variant", gr.Radio, {"choices": ["tab", "rhoab"]}, infotext='DEIS variant'),
    'deis_order': OptionInfo(3, "DEIS order", gr.Slider, {"minimum": 2, "maximum": 4, "step": 1}, infotext='DEIS order').info("must be < sampling steps"),

    'dpmpp_2m_sde_mode': OptionInfo("midpoint", "DPM++ 2M SDE variant", gr.Radio, {"choices": ["heun", "midpoint"]}, infotext="2M SDE variant"),

    'lcm_order': OptionInfo(1, "LCM order", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, infotext='LCM order').info("limited by number of sampling steps"),

    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}, infotext='UniPC variant'),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}, infotext='UniPC skip type'),
    'uni_pc_order': OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, infotext='UniPC order').info("must be < sampling steps"),
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final", infotext='UniPC lower order final'),

    'sd_noise_schedule': OptionInfo("Default", "Noise schedule for sampling", gr.Radio, {"choices": ["Default", "Zero Terminal SNR"]}, infotext="Noise Schedule").info("for use with zero terminal SNR trained models"),
    'skip_early_cond': OptionInfo(0.0, "Ignore negative prompt during early sampling", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Skip Early CFG").info("disables CFG on a proportion of steps at the beginning of generation; 0=skip none; 1=skip all; can both improve sample diversity/quality and speed up sampling"),
    'beta_dist_alpha': OptionInfo(0.6, "Beta scheduler - alpha", gr.Slider, {"minimum": 0.01, "maximum": 1.0, "step": 0.01}, infotext='Beta alpha').info('Default = 0.6; the alpha parameter of the beta distribution used in Beta sampling'),
    'beta_dist_beta': OptionInfo(0.6, "Beta scheduler - beta", gr.Slider, {"minimum": 0.01, "maximum": 1.0, "step": 0.01}, infotext='Beta beta').info('Default = 0.6; the beta parameter of the beta distribution used in Beta sampling'),
    'sigmoid_base_c': OptionInfo(0.5, "Sigmoid offset scheduler - base c", gr.Slider, {"minimum": -50.0, "maximum": 50.0, "step": 0.01}, infotext='base c').info('Default = 0.5; the base c parameter of the Sigmoid offset scheduler sampling'),
    'sigmoid_square_k': OptionInfo(1.0, "Sigmoid offset scheduler - square k", gr.Slider, {"minimum": 0.01, "maximum": 10.0, "step": 0.01}, infotext='square k').info('Default = 1.0; the square k parameter of the Sigmoid offset scheduler sampling'),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing", "postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_disable_in_extras': OptionInfo([], "Disable postprocessing operations in extras tab", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
}))

forge_shared_options.register(options_templates, options_section, OptionInfo)



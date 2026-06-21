import dataclasses
import json
import html
import os

import gradio as gr

from modules import shared, ui_tempdir, util
import modules.images
from modules.ui_components import ToolButton
import modules.infotext_utils as parameters_copypaste

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄


def select_gallery_and_update_gen_info(generation_info, html_info, index):
    if index < 0:
        return gr.update(selected_index=0), update_generation_info(generation_info, html_info, 0)

    return gr.skip(), update_generation_info(generation_info, html_info, index)


def update_generation_info(generation_info, html_info, index):
    gen_info = html_info
    try:
        generation_info = json.loads(generation_info)
        if index >= 0 and index < len(generation_info["infotexts"]):
            gen_info = plaintext_to_html(generation_info["infotexts"][index])
    except Exception:
        pass

    return gen_info


def plaintext_to_html(text, classname=None):
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"


@dataclasses.dataclass
class OutputPanel:
    gallery = None
    generation_info = None
    infotext = None
    html_log = None
    button_upscale = None


def create_output_panel(tabname, outdir, toprow=None):  # used by txt2img, img2img, extras
    res = OutputPanel()

    def open_folder(filename):
        if shared.cmd_opts.hide_ui_dir_config:
            return

        f = shared.opts.outdir_samples or outdir

        if filename.startswith("http://") and "/file=" in filename:
            filename = filename.split("/file=", 1)[1]
            if os.path.exists(filename):
                if "Subdirectory" in shared.opts.open_dir_button_choice:
                    image_dir = os.path.split(filename)[0]

                    if "temp" in shared.opts.open_dir_button_choice or not ui_tempdir.is_gradio_temp_path(image_dir):
                        f = image_dir

        util.open_folder(f)

    with gr.Column(elem_id=f"{tabname}_results"):
        if toprow:
            toprow.submit_box.render()

        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            dummy = gr.Number(value=0, visible=False)

            res.gallery = gr.Gallery(label="Output", show_label=False, elem_id=f"{tabname}_gallery", columns=4, preview=True, height=shared.opts.gallery_height or None, interactive=False, type="pil", object_fit="contain")


        with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
            open_folder_button = ToolButton(folder_symbol, elem_id=f"{tabname}_open_folder", visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open images output directory.")

            buttons = {
                "img2img": ToolButton("🖼️", elem_id=f"{tabname}_send_to_img2img", tooltip="Send image and generation parameters to img2img tab."),
                "extras": ToolButton("📐", elem_id=f"{tabname}_send_to_extras", tooltip="Send image and generation parameters to extras tab.")
            }

            if tabname == "txt2img":
                toprow.button_upscale.render()

        open_folder_button.click(
            fn=open_folder,
            js="selected_gallery_button_filename",
            inputs=[dummy],
            outputs=None,
        )

        if tabname == "extras":
            res.generation_info = gr.HTML(elem_id=f"html_info_{tabname}")
        else:
            res.generation_info = gr.Textbox(visible=False, elem_id=f"generation_info_{tabname}")
            res.infotext = gr.HTML(elem_id=f"html_info_{tabname}", elem_classes="infotext")
            res.gallery.select(fn=update_generation_info, js="function(x, y, z){ return [x, y, selected_gallery_index()] }", inputs=[res.generation_info, res.infotext, dummy], outputs=[res.infotext], show_progress="hidden")

        res.gallery.change(fn=None, js="setup_gallery_lightbox", inputs=None, outputs=None)

        res.html_log = gr.HTML(elem_id=f"html_log_{tabname}", elem_classes="html-log")

        paste_field_names = []
        if tabname == "txt2img":
            paste_field_names = modules.scripts.scripts_txt2img.paste_field_names
        elif tabname == "img2img":
            paste_field_names = modules.scripts.scripts_img2img.paste_field_names

        for paste_tabname, paste_button in buttons.items():
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=paste_button, tabname=paste_tabname, source_tabname="txt2img" if tabname == "txt2img" else None, source_image_component=res.gallery,
                paste_field_names=paste_field_names
            ))

    return res


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, "label", None)
        if label is not None:
            break

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id, tooltip=f"{label}: refresh" if label else "Refresh")
    refresh_button.click(
        fn=refresh,
        inputs=None,
        outputs=refresh_components
    )
    return refresh_button


def setup_dialog(button_show, dialog, *, button_close=None):
    """Sets up the UI so that the dialog (gr.Box) is invisible, and is only shown when buttons_show is clicked, in a fullscreen modal window."""

    dialog.visible = False

    button_show.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=[dialog],
    ).then(fn=None, js="function(){ popupId('" + dialog.elem_id + "'); }")

    if button_close:
        button_close.click(fn=None, js="closePopup")


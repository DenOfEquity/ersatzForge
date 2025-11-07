import gradio as gr

from modules import shared, ui_common, ui_components, styles

styles_edit_symbol = '\U0001f58c\uFE0F'  # 🖌️
styles_materialize_symbol = '\U0001f4cb'  # 📋
styles_copy_symbol = '\U0001f4dd'  # 📝


def select_style(name):
    style = shared.prompt_styles.styles.get(name)
    existing = style is not None
    empty = not name

    prompt = style.prompt if style else gr.skip()
    negative_prompt = style.negative_prompt if style else gr.skip()

    return prompt, negative_prompt, gr.update(interactive=existing), gr.update(interactive=not empty)


def save_style(name, prompt, negative_prompt):
    if not name:
        return gr.update(interactive=False)

    existing_style = shared.prompt_styles.styles.get(name)
    path = existing_style.path if existing_style is not None else None

    style = styles.PromptStyle(name, prompt, negative_prompt, path)
    shared.prompt_styles.styles[style.name] = style
    shared.prompt_styles.save_styles()

    return gr.update(interactive=True)


def delete_style(name):
    if name == "":
        return

    shared.prompt_styles.styles.pop(name, None)
    shared.prompt_styles.save_styles()

    return '', '', ''


def materialize_styles(prompt, negative_prompt, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(negative_prompt, styles)

    return [gr.update(value=prompt), gr.update(value=negative_prompt), gr.update(value=[])]


def refresh_styles():
    return gr.update(choices=list(shared.prompt_styles.styles)), gr.update(choices=list(shared.prompt_styles.styles))


class UiPromptStyles:
    def __init__(self, tabname, main_ui_prompt, main_ui_negative_prompt):
        self.tabname = tabname
        self.main_ui_prompt = main_ui_prompt
        self.main_ui_negative_prompt = main_ui_negative_prompt

        # with gr.Row(elem_id=f"{tabname}_styles_row"):
        apply_button = ui_components.ToolButton(value=styles_materialize_symbol, elem_id=f"{tabname}_style_apply", tooltip="Apply all selected styles to prompts. Strips comments, if enabled.")
        self.dropdown = gr.Dropdown(label="Styles", show_label=False, elem_id=f"{tabname}_styles", choices=list(shared.prompt_styles.styles), value=[], multiselect=True, tooltip="Styles")
        edit_button = ui_components.ToolButton(value=styles_edit_symbol, elem_id=f"{tabname}_styles_edit_button", tooltip="Edit styles")

        with gr.Group(elem_id=f"{tabname}_styles_dialog", elem_classes="popup-dialog") as styles_dialog:
            gr.Markdown('''
## Styles Editor ##
Styles are additions to the **Prompt** and **Negative prompt**. The *{prompt}* token, if it exists, will be replaced with the user prompt when the style is applied. Otherwise, the style text will be added to the end of the prompt.
''')
            with gr.Row():
                self.copy = ui_components.ToolButton(value=styles_copy_symbol, elem_id=f"{tabname}_style_copy", tooltip="Copy main UI prompt to style.")
                self.selection = gr.Dropdown(show_label=False, elem_id=f"{tabname}_styles_edit_select", choices=list(shared.prompt_styles.styles), value=[], allow_custom_value=True)
                ui_common.create_refresh_button([self.dropdown, self.selection], shared.prompt_styles.reload, lambda: {"choices": list(shared.prompt_styles.styles)}, f"refresh_{tabname}_styles")

                self.delete = gr.Button('Delete', variant='primary', scale=0, interactive=False, elem_classes=["leftpadbutton"])
                self.save   = gr.Button('Save',   variant='primary', scale=0, interactive=False, elem_classes=["leftpadbutton"])

            self.prompt = gr.Textbox(label="Prompt", show_label=True, elem_id=f"{tabname}_edit_style_prompt", lines=3, elem_classes=["prompt"])
            self.neg_prompt = gr.Textbox(label="Negative prompt", show_label=True, elem_id=f"{tabname}_edit_style_neg_prompt", lines=3, elem_classes=["prompt"])


        self.selection.change(
            fn=select_style,
            inputs=[self.selection],
            outputs=[self.prompt, self.neg_prompt, self.delete, self.save],
            show_progress=False,
        )

        self.save.click(
            fn=save_style,
            inputs=[self.selection, self.prompt, self.neg_prompt],
            outputs=[self.delete],
            show_progress=False,
        ).then(refresh_styles, outputs=[self.dropdown, self.selection], show_progress=False)

        self.delete.click(
            fn=delete_style,
            js='function(name){ if(name == "") return ""; return confirm("Delete style " + name + "?") ? name : ""; }',
            inputs=[self.selection],
            outputs=[self.selection, self.prompt, self.neg_prompt],
            show_progress=False,
        ).then(refresh_styles, outputs=[self.dropdown, self.selection], show_progress=False)

        self.copy.click(
            fn=lambda p, n: (p, n),
            inputs=[main_ui_prompt, main_ui_negative_prompt],
            outputs=[self.prompt, self.neg_prompt],
            show_progress=False,
        )

        ui_common.setup_dialog(button_show=edit_button, dialog=styles_dialog, button_close=None)

        apply_button.click(
            fn=materialize_styles,
            inputs=[main_ui_prompt, main_ui_negative_prompt, self.dropdown],
            outputs=[main_ui_prompt, main_ui_negative_prompt, self.dropdown],
            show_progress=False,
        )

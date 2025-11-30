import gradio as gr

from modules import shared, ui_prompt_styles
import modules.images

from modules.ui_components import ToolButton


class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    prompt = None
    prompt_img = None
    negative_prompt = None

    button_deepbooru = None

    interrupt = None
    skip = None
    submit = None

    paste = None

    if not shared.opts.disable_token_counters:
        token_counter = None
        token_button = None
        negative_token_counter = None
        negative_token_button = None

    ui_styles = None

    submit_box = None

    def __init__(self, id_part):
        self.id_part = id_part

        self.create_submit_box()

    def create_inline_toprow_prompts(self):
        self.create_prompts()

        with gr.Row(elem_classes=["toprow-compact-stylerow"]):
            self.create_tools_row()
            self.create_styles_ui()

    def create_prompts(self):
        with gr.Column(elem_id=f"{self.id_part}_prompt_container", elem_classes=["prompt-container-compact"], scale=6):
            with gr.Row(elem_id=f"{self.id_part}_prompt_row", elem_classes=["prompt-row"]):
                self.prompt = gr.Textbox(label="Prompt", elem_id=f"{self.id_part}_prompt", show_label=False, lines=3, placeholder="Prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value='')
                self.prompt_img = gr.File(label="", elem_id=f"{self.id_part}_prompt_image", file_count="single", type="binary", visible=False)

            with gr.Row(elem_id=f"{self.id_part}_neg_prompt_row", elem_classes=["prompt-row"]):
                self.negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{self.id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value='')

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress="hidden",
        )

    def create_submit_box(self):
        with gr.Row(elem_id=f"{self.id_part}_generate_box", elem_classes=["generate-box"] + (["generate-box-compact"]), render=False) as submit_box:
            self.submit_box = submit_box
            if self.id_part == "extras":
                shared.opts.extras_save = True
                autosave_extras = ToolButton(value="💾", variant="primary" if shared.opts.extras_save else "secondary")

                def toggleAutosaveExtras():
                    shared.opts.extras_save ^= True
                    print (f"[Extras] Autosave result(s): {shared.opts.extras_save}")
                    return gr.update(variant="primary" if shared.opts.extras_save else "secondary")

                autosave_extras.click(fn=toggleAutosaveExtras, inputs=None, outputs=[autosave_extras])

            self.interrupt = gr.Button('Interrupt', elem_id=f"{self.id_part}_interrupt", elem_classes="generate-box-interrupt", tooltip="End generation")
            self.skip = gr.Button('Skip', elem_id=f"{self.id_part}_skip", elem_classes="generate-box-skip", tooltip="Stop generation of current batch and continues onto next batch")
            self.submit = gr.Button('Generate', elem_id=f"{self.id_part}_generate", variant='primary', tooltip="Right click generate forever menu" if self.id_part != "extras" else None)

            self.skip.click(fn=shared.state.skip)
            self.interrupt.click(fn=shared.state.interrupt, js='function(){ showSubmitInterruptingPlaceholder("' + self.id_part + '"); }')

    def create_tools_row(self):
        paste_symbol = '\u2199\ufe0f'  # ↙

        self.paste = ToolButton(value=paste_symbol, elem_id="paste", tooltip="Read generation parameters from prompt or last generation if prompt is empty into user interface.")

        if self.id_part == "img2img":
            self.button_deepbooru = ToolButton('📦', tooltip='Interrogate DeepBooru - use DeepBooru neural network to describe the image, and put it into the Prompt field', elem_id="deepbooru")

        if not shared.opts.disable_token_counters:
            self.token_counter = gr.HTML(value="<span>1/75</span>", elem_id=f"{self.id_part}_token_counter", elem_classes=["token-counter"])
            self.token_button = gr.Button(visible=False, elem_id=f"{self.id_part}_token_button")
            self.negative_token_counter = gr.HTML(value="<span>1/75</span>", elem_id=f"{self.id_part}_negative_token_counter", elem_classes=["token-counter"])
            self.negative_token_button = gr.Button(visible=False, elem_id=f"{self.id_part}_negative_token_button")


    def create_styles_ui(self):
        self.ui_styles = ui_prompt_styles.UiPromptStyles(self.id_part, self.prompt, self.negative_prompt)

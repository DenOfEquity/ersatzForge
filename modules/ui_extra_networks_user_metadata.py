import datetime
import html
import json
import os.path

import gradio as gr

from modules import infotext_utils, images, sysinfo, errors, ui_extra_networks, shared


class UserMetadataEditor:

    def __init__(self, ui, tabname, page):
        self.ui = ui
        self.tabname = tabname
        self.page = page
        self.id_part = f"{self.tabname}_{self.page.extra_networks_tabname}_edit_user_metadata"

        self.box = None

        self.edit_name_input = None
        self.button_edit = None

        self.edit_name = None
        self.edit_description = None
        self.edit_notes = None
        self.html_filedata = None
        self.html_preview = None

        # self.button_cancel = None
        self.button_replace_preview = None
        self.button_save = None

    def get_user_metadata(self, name):
        item = self.page.items.get(name, {})

        user_metadata = item.get('user_metadata', None)
        if not user_metadata:
            user_metadata = {'description': item.get('description', '')}
            item['user_metadata'] = user_metadata

        return user_metadata

    def create_extra_default_items_in_left_column(self):
        pass

    def create_default_editor_elems(self):
        with gr.Row():
            with gr.Column(scale=2):
                self.edit_name = gr.Markdown()
                self.edit_description = gr.Textbox(label="Description", lines=4)
                self.html_filedata = gr.Markdown()

                self.create_extra_default_items_in_left_column()

            with gr.Column(scale=1, min_width=0):
                self.html_preview = gr.HTML()

    def create_default_buttons(self):

        with gr.Row(elem_classes="edit-user-metadata-buttons"):
            # self.button_cancel = gr.Button('Cancel')
            self.button_replace_preview = gr.Button('Replace preview', variant='primary')
            self.button_save = gr.Button('Save', variant='primary')

        # self.button_cancel.click(fn=None, js="closePopup")

    def get_card_html(self, name):
        item = self.page.items.get(name, {})

        preview_url = item.get("preview", None)

        if not preview_url:
            filename, _ = os.path.splitext(item["filename"])
            preview_url = self.page.find_preview(filename)
            item["preview"] = preview_url

        if preview_url:
            preview = f'''
            <div class='card standalone-card-preview'>
                <img src="{html.escape(preview_url)}" class="preview">
            </div>
            '''
        else:
            preview = "<div class='card standalone-card-preview'></div>"

        return preview

    def relative_path(self, path):
        for parent_path in self.page.allowed_directories_for_previews():
            if ui_extra_networks.path_is_parent(parent_path, path):
                return os.path.relpath(path, parent_path)

        return os.path.basename(path)

    def get_metadata_table(self, name):
        item = self.page.items.get(name, {})
        try:
            filename = item["filename"]
            shorthash = item.get("shorthash", None)

            stats = os.stat(filename)
            params = [
                ('File path', filename),
                ('File size', sysinfo.pretty_bytes(stats.st_size)),
                ('Hash', shorthash),
                ('Modified', datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')),
            ]

            return params
        except Exception as e:
            errors.display(e, f"reading info for {name}")
            return []

    def put_values_into_components(self, name):
        user_metadata = self.get_user_metadata(name)

        try:
            params = self.get_metadata_table(name)
        except Exception as e:
            errors.display(e, f"reading metadata info for {name}")
            params = []

        table = "| *metadata* | *value* | \n|---|---|\n" + "\n".join(f"| {name} | {html.escape(str(value))} |" for name, value in params if value is not None)

        return f'## {name}', user_metadata.get('description', ''), table, self.get_card_html(name), user_metadata.get('notes', '')

    def write_user_metadata(self, name, metadata):
        item = self.page.items.get(name, {})
        filename = item.get("filename", None)
        basename, ext = os.path.splitext(filename)

        metadata_path = basename + '.json'
        with open(metadata_path, "w", encoding="utf8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)
        self.page.lister.update_file_entry(metadata_path)

    def save_user_metadata(self, name, desc, notes):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["notes"] = notes

        self.write_user_metadata(name, user_metadata)

    def setup_save_handler(self, button, func, components):
        button\
            .click(fn=func, inputs=[self.edit_name_input, *components], outputs=None, show_progress=False)\
            .then(fn=None, js="function(name){closePopup(); extraNetworksRefreshSingleCard(" + json.dumps(self.page.name) + "," + json.dumps(self.tabname) + ", name);}", inputs=[self.edit_name_input], outputs=None, show_progress=False)

    def create_editor(self):
        self.create_default_editor_elems()

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        self.create_default_buttons()

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=[self.edit_name, self.edit_description, self.html_filedata, self.html_preview, self.edit_notes], show_progress=False)\
            .then(fn=lambda: gr.update(visible=True), inputs=None, outputs=[self.box], show_progress=False)

        self.setup_save_handler(self.button_save, self.save_user_metadata, [self.edit_description, self.edit_notes])

    def create_ui(self):
        with gr.Box(visible=False, elem_id=self.id_part, elem_classes="edit-user-metadata") as box:
            self.box = box

            self.edit_name_input = gr.Textbox("Edit user metadata card id", visible=False, elem_id=f"{self.id_part}_name")
            self.button_edit = gr.Button("Edit user metadata", visible=False, elem_id=f"{self.id_part}_button")

            self.create_editor()

    def save_preview(self, index, gallery, name):
        if not gallery or len(gallery) == 0:
            gr.Info("No gallery: preview image not changed.", 5)
            return self.get_card_html(name)

        item = self.page.items.get(name, {})

        index = int(index)
        index = max(index, 0)
        index = min(index, len(gallery)-1)

        img_info = gallery[index]
        image = infotext_utils.image_from_url_text(img_info)
        geninfo, items = images.read_info_from_image(image)

        tw = 2 * shared.opts.extra_networks_card_width
        th = 2 * shared.opts.extra_networks_card_height
        old_aspect = image.size[0] / image.size[1]
        new_aspect = tw / th

        if old_aspect > new_aspect: # original image is wider than needed, adjust preview to match
            rw = int(old_aspect * th)
            rh = th
            crop = ((rw - tw)//2, 0, (rw + tw)//2, th)
        else:
            rw = tw
            rh = int(tw / old_aspect)
            crop = (0, (rh - th)//2, tw, (rh + th)//2)
        
        images.save_image_with_geninfo(image.resize((rw, rh)).crop(crop), geninfo, item["local_preview"])
        self.page.lister.update_file_entry(item["local_preview"])
        item['preview'] = self.page.find_preview(item["local_preview"])
        return self.get_card_html(name)

    def setup_ui(self, gallery):
        self.button_replace_preview.click(
            fn=self.save_preview,
            js=f"function(x, y, z){{return [selected_gallery_index_id('{self.tabname + '_gallery_container'}'), y, z]}}",
            inputs=[self.edit_name_input, gallery, self.edit_name_input],
            outputs=[self.html_preview],
            show_progress=False
        ).then(
            fn=None,
            js="function(name){extraNetworksRefreshSingleCard(" + json.dumps(self.page.name) + "," + json.dumps(self.tabname) + ", name);}",
            inputs=[self.edit_name_input],
            outputs=None,
            show_progress=False
        )

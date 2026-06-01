import gradio as gr

from modules import processing, shared, images
from modules.processing import Processed
from modules.shared import opts, state
import modules.scripts as scripts


class Script(scripts.Script):
    def title(self):
        return "SD Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row():
            width = gr.Slider(label="Tile width", value=512, minimum=256, maximum=2048, step=64)
            height = gr.Slider(label="Tile height", value=512, minimum=256, maximum=2048, step=64)
        with gr.Row():
            overlap = gr.Slider(label="Tile overlap", value=64, minimum=0, maximum=256, step=16)
            scale = gr.Slider(label="Upscale factor", value=2.0, minimum=1.0, maximum=4.0, step=0.05)
        upscaler_index = gr.Radio(label="Upscaler", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[1].name, type="index")

        return [width, height, overlap, scale, upscaler_index]

    def run(self, p, width, height, overlap, scale, upscaler_index):
        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(upscaler_index.lower())
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]

        p.extra_generation_params["SD Upscale"] = (scale, upscaler.name, width, height, overlap)

        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        if upscaler.name != "None":
            img = upscaler.scaler.upscale(init_img, scale, upscaler.data_path)
        else:
            img = init_img

        grid = images.split_grid(img, tile_w=width, tile_h=height, overlap=overlap)

        batch_size = p.batch_size
        upscale_count = p.n_iter
        original_width = p.width
        original_height = p.height
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.width = width
        p.height = height

        work = []

        for _y, _h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = int((len(work) + batch_size - 1) / batch_size)
        state.job_count = batch_count * upscale_count

        print(f"[SD Upscale] {upscale_count} output images; {len(work)*upscale_count} total tiles ({len(grid.tiles[0][2])}x{len(grid.tiles)} per image); {state.job_count} total batches.")

        result_images = []
        infotexts = []
        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed

            work_results = []
            for i in range(batch_count):
                p.batch_size = batch_size
                p.init_images = work[i * batch_size:(i + 1) * batch_size]

                state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                processed = processing.process_images(p)

                if i == 0:
                    infotexts.append(processed.info.replace(f"Size: {width}x{height}", f"Size: {original_width}x{original_height}", 1))

                p.seed = processed.seed + 1
                work_results += processed.images

            if not (state.interrupted or state.stopping_generation):
                image_index = 0
                for _y, _h, row in grid.tiles:
                    for tiledata in row:
                        tiledata[2] = work_results[image_index]
                        image_index += 1

                combined_image = images.combine_grid(grid)
                result_images.append(combined_image)

                if opts.samples_save:
                    images.save_image(combined_image, p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=infotexts[-1], p=p)

        processed = Processed(p, result_images, seed, "", infotexts=infotexts)

        p.n_iter = upscale_count

        return processed

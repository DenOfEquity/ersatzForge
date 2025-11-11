import copy
import random
import gradio

from modules import errors, sd_models, scripts
from modules.processing import Processed, process_images
from modules.shared import opts, state
from modules.images import image_grid, save_image


def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    assert info is not None, f'Unknown checkpoint: {tag}'
    return info.name


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag.lower() == "true") else False


def process_size_tag(tag):
    if tag[0] == "r":
        base = int(tag[1:])
        lo = 3 * base // 4
        hi = 5 * base // 3
        w = int(random.randrange(lo, hi, 16))
        h = int(base * base / w)
        h = 16 * ((h + 8) // 16)
    elif "x" in tag:
        w, h = tag.split("x", 1)
        w, h = int(w), int(h)
    else:
        w = h = int(tag)

    return w, h


prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "distilled_cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_string_tag,
    "do_not_save_samples": process_boolean_tag,
}


def cmdargs(line):
    args = line.split(" ")
    pos = 0
    res = {}

    def get_full_text(pos, args):
        pos_e = pos
        while pos_e < len(args) and not args[pos_e].startswith("--"):
            pos_e += 1

        text = " ".join(args[pos:pos_e])

        return pos_e, text

    while pos < len(args):
        arg = args[pos]

        if arg == "":
            pos += 1
            continue
        if not arg.startswith("--"):
            print (f'[Prompts from file] argument must start with "--": {arg}')
            pos += 2
            continue
        if pos+1 >= len(args):
            print (f'[Prompts from file] missing data for argument {arg}')
            pos += 2
            continue

        tag = arg[2:].lower()

        if tag in ["prompt", "negative_prompt", "sampler_name", "scheduler"]:
            pos, res[tag] = get_full_text(pos+1, args)
            continue

        if tag == "size":
            val = args[pos+1]
            res["width"], res["height"] = process_size_tag(val)
        elif func := prompt_tags.get(tag, None):
            val = args[pos+1]
            res[tag] = func(val)

        pos += 2

    return res


def load_prompt_file(file):
    if file is None:
        return None, gradio.skip()
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        return None, "\n".join(lines)


class Script(scripts.Script):
    def title(self):
        return "Prompts from file or textbox"

    def ui(self, is_img2img):
        checkbox_iterate = gradio.Radio(["None", "Iterate every line", "Same every line"], label="Seed iteration", value="None")
        prompt_position = gradio.Radio(["Start", "End"], label="Insert prompts at the", value="Start")
        make_combined = gradio.Checkbox(label="Make a combined image containing all outputs (if more than one)", value=False)

        prompt_txt = gradio.Textbox(label="List of prompt inputs", lines=2)
        prompt_file = gradio.File(label="Upload prompt inputs", type='binary')

        prompt_file.upload(fn=load_prompt_file, inputs=[prompt_file], outputs=[prompt_file, prompt_txt], show_progress=False)

        return [checkbox_iterate, prompt_position, prompt_txt, make_combined]

    def run(self, p, checkbox_iterate, prompt_position, prompt_txt: str, make_combined):
        lines = [x for x in (x.strip() for x in prompt_txt.splitlines()) if x]

        p.do_not_save_grid = True
        p.fill_fields_from_opts()

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    errors.report(f"Error parsing line {line} as commandline", exc_info=True)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        print (f"[Prompts from file] will process {len(lines)} lines in {job_count} jobs.")
        if checkbox_iterate != "None" and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                if k == "sd_model":
                    copy_p.override_settings['sd_model_checkpoint'] = v
                else:
                    setattr(copy_p, k, v)

            if args.get("prompt") and p.prompt:
                if prompt_position == "Start":
                    copy_p.prompt = args.get("prompt") + " " + p.prompt
                else:
                    copy_p.prompt = p.prompt + " " + args.get("prompt")

            if args.get("negative_prompt") and p.negative_prompt:
                if prompt_position == "Start":
                    copy_p.negative_prompt = args.get("negative_prompt") + " " + p.negative_prompt
                else:
                    copy_p.negative_prompt = p.negative_prompt + " " + args.get("negative_prompt")

            proc = process_images(copy_p)
            images += proc.images

            if checkbox_iterate == "Iterate every line":
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        if make_combined and len(images) > 1:
            combined_image = image_grid(images, batch_size=1, rows=None).convert("RGB")
            full_infotext = "\n".join(infotexts)

            is_img2img = getattr(p, "init_images", None) is not None

            if opts.grid_save:  #   use grid specific Settings
                save_image(
                    combined_image,
                    opts.outdir_grids or (opts.outdir_img2img_grids if is_img2img else opts.outdir_txt2img_grids),
                    "",
                    -1,
                    prompt_txt,
                    opts.grid_format,
                    full_infotext,
                    grid=True
                )
            else:               #   use normal output Settings
                save_image(
                    combined_image,
                    opts.outdir_samples or (opts.outdir_img2img_samples if is_img2img else opts.outdir_txt2img_samples),
                    "",
                    -1,
                    prompt_txt,
                    opts.samples_format,
                    full_infotext
                )

            images.insert(0, combined_image)
            all_prompts.insert(0, prompt_txt)
            infotexts.insert(0, full_infotext)

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)

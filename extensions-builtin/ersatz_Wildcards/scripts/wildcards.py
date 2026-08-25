# modified from github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards/

# find a better id than '__'? : why is interrobang hard to type ‽

import os
import random

from modules import scripts, script_callbacks, shared

wildcards = {}
variables = {}
indices = {}

class WildcardsScript(scripts.Script):
    def __init__(self):
        if wildcards == {}: # preload wildcard files: avoids loading/split/de-dup files each run
            self.load_wildcards()

    def title(self):
        return "DoE Ersatz wildcards"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def load_wildcards(self):
        wildcards_dir = os.path.join(scripts.basedir(), "wildcards")
        cut = len(wildcards_dir) + 1
        for filepath in shared.walk_files(wildcards_dir, allowed_extensions=[".txt"]):
            name = filepath[cut:-4].replace("\\", "/")
            with open(filepath, encoding="utf8") as f:
                choices = [x for x in f.read().splitlines() if not x.startswith("#")]
                wildcards[name] = sorted(set(choices))  # remove duplicates; and sort, otherwise order is random
                indices[name] = [0, len(wildcards[name]), ""]

    def replace_wildcard(self, text, gen):
        if len(text) <= 4:
            return text

        while True: # will replace multiple wildcards in one 'word'
            s_index = text.find("__")
            if s_index == -1:
                return text

            e_index = text.find("__", s_index+2)
            if e_index == -1:
                return text

            prefix   = text[:s_index]
            wildcard = text[s_index+2:e_index]
            suffix   = text[e_index+2:]

            if wildcard in variables:
                wildcard = gen.choice(variables[wildcard])
            elif wildcard[0] == "@" and wildcard[1:] in wildcards:
                w = wildcard[1:]
                wildcard = wildcards[w][indices[w][0]]
                indices[w][0] = (indices[w][0] + 1) % indices[w][1]
                indices[w][2] = wildcard
            elif wildcard[0] == "<" and wildcard[1:] in wildcards:
                w = wildcard[1:]
                wildcard = indices[w][2]
            elif wildcard in wildcards:
                w = wildcard
                wildcard = gen.choice(wildcards[wildcard])
                indices[w][2] = wildcard

            text = prefix + wildcard + suffix

    def apply_wildcards(self, p, attr, infotext_suffix, compare):
        if original_prompts := getattr(p, attr, None):
            result = []

            for i, prompt in enumerate(original_prompts):
                if prompt.startswith("!L"):
                    p.all_seeds[i] = p.all_seeds[0]
                    prompt = prompt[2:]

                gen = random.Random()
                gen.seed(p.all_seeds[i])

                # load directly into prompt, allows wildcard files to include commands
                s_index = prompt.find("!$__")
                while s_index != -1:
                    e_index = prompt.find("__", s_index+4)
                    if e_index == -1:
                        break
                    wildcard = prompt[s_index+4:e_index]
                    if wildcard in wildcards:
                        prefix = prompt[:s_index]
                        suffix = prompt[e_index+2:]
                        prompt = prefix + gen.choice(wildcards[wildcard]) + suffix
                        s_index = prompt.find("!$__", len(prefix))
                    else:
                        s_index = e_index+2

                new_prompt = []
                for text in prompt.split():
                    method = None

                    if text.startswith("!") and "=" in text:         # variable SET
                        variable, text = text[1:].split("=", 2)
                        method = "Set"
                    elif text.startswith("!") and "+" in text:         # variable EXTEND - useful?
                        variable, text = text[1:].split("+", 2)
                        method = "Extend" if variable in variables else "Set"
                    elif text.startswith("!") and "&" in text:         # variable APPEND / another option - useful?
                        variable, text = text[1:].split("&", 2)
                        method = "Append" if variable in variables else "Set"

                    if text.startswith("!") and ":" in text:         # SELECT
                        text = gen.choice(text[1:].split(":"))

                    if text.startswith("!") and "*" in text:         # REPEAT
                        count, var = text[1:].split("*", 2)
                        text = " ".join([var] * int(count))

                    text = self.replace_wildcard(text.replace("^", " "), gen)

                    if method == "Set": # SET can EXTEND and APPEND in one
                        variables[variable] = text.replace("+", " ").split("&")#[text]
                    elif method == "Append":
                        variables[variable].append(text)
                    elif method == "Extend":
                        variables[variable] = [x + " " + text for x in variables[variable]]
                    else:
                        new_prompt.append(text)

                result.append(" ".join(new_prompt))

            if compare is not None: # HiRes (negative) prompt
                compare_prompts = getattr(p, compare)
                for i in range(len(result)): # this command processing is after wildcard substitution, seems more correct
                    if result[i].startswith("!ADD_START "):
                        result[i] = " ".join([result[i][11:], compare_prompts[i]])
                    elif result[i].startswith("!ADD_END "):
                        result[i] = " ".join([compare_prompts[i], result[i][9:]])

            if result[0].split() != original_prompts[0].split():
                setattr(p, attr, result)
                if compare is None or result[0].split() != compare_prompts[0].split():
                    p.extra_generation_params[f"Wildcard {infotext_suffix}"] = original_prompts[0]

    def process(self, p, *args):
        if getattr(shared.opts, "wildcards_reload", False):
            wildcards.clear()
            self.load_wildcards()

        if wildcards != {}:
            variables.clear()
            for k, _v in indices.items():
                indices[k][0] = 0
                indices[k][2] = ""

            self.apply_wildcards(p, "all_prompts",              "prompt",                None)
            self.apply_wildcards(p, "all_negative_prompts",     "negative prompt",       None)
            self.apply_wildcards(p, "all_hr_prompts",           "hr prompt",            "all_prompts")
            self.apply_wildcards(p, "all_hr_negative_prompts",  "hr negative prompt",   "all_negative_prompts")


def on_ui_settings():
    shared.opts.add_option("wildcards_reload", shared.OptionInfo(False, "[Ersatz wildcards] Load wildcards >> Disabled: Load wildcards only on UI start. Enabled: Reload wildcards every run.", section=("wildcards", "Wildcards")))

script_callbacks.on_ui_settings(on_ui_settings)

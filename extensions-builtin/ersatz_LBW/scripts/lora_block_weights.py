# cut down from https://github.com/hako-mikan/sd-webui-lora-block-weight

import os
import re
import importlib
import gradio as gr
import os.path
import random
import time
from modules import sd_models, extra_networks, scripts, shared
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser, remove_current_script_callbacks
from modules.ui_components import InputAccordion


BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61
BLOCKIDZIT = ["J{:002}".format(x) for x in range(30)]
#todo chroma
BLOCKNUMS = [12, 17, 20, 26, len(BLOCKIDZIT), len(BLOCKIDFLUX)]
BLOCKIDS=[BLOCKID12, BLOCKID17, BLOCKID20, BLOCKID26, BLOCKIDZIT, BLOCKIDFLUX]

BLOCKS=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders",
"transformer_resblocks"]


def reloadPresets():
    elements = []
    weights = []
    file = os.path.abspath(__file__[:-2] + "txt")
    with open(file, 'r') as f:
        text = f.read().splitlines()
    for line in text:
        if line.startswith("WEIGHT:"):
            weights.append(line[7:].split(":", 1))
        elif line.startswith("ELEMENT:"):
            elements.append(line[8:].split(":"))
    return weights, elements

weights, elements = reloadPresets()


class ersatzLBW(scripts.Script):
    def __init__(self):
        self.lratios = {}
        self.elementals = {}

        self.stopsf = []
        self.startsf = []
        self.uf = []
        self.lf = []
        self.ef = []

    def title(self):
        return "LoRA Block Weight Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown("Block weights presets, :lbw=")
            lbw_loraratios = gr.Dataframe(show_label=False, headers=["Name", "Blocks"], col_count=(2, 'fixed'), column_widths=["15%", "75%"], type="array", value=weights, wrap=True, interactive=True, height=240)      
            gr.Markdown("Elements presets: **Identifer:BlockID:Element:Weight**, :lbwe=")
            lbw_elementals = gr.Dataframe(show_label=False, headers=["Name", "Blocks", "Element", "Weight"], col_count=(4, 'fixed'), column_widths=["20%", "40%", "18%", "12%"], type="array", value=elements, wrap=True, interactive=True, height=240)      
            gr.Markdown("*additions to LoRA prompt format*: :start=**STEP** :stop=**STEP** *(actual step, or proportional)*")
            with gr.Row():
                reload = gr.Button("Reload", scale=0, elem_classes=["LBW_button"])
                gr.Markdown("[*original extension by* **hako-mikan**](https://github.com/hako-mikan/sd-webui-lora-block-weight)")
                save = gr.Button("Save", scale=0, elem_classes=["LBW_button"])

                def savePresets(ratios, elements):
                    text = ""
                    for ratio in ratios:
                        if ratio[0].strip() != "" and len(ratio[1].split(",")) in BLOCKNUMS:
                            text += f"WEIGHT:{ratio[0].strip()}:{ratio[1]}\n"
                    for element in elements:
                        if element[0].strip() != "":
                            text += f"ELEMENT:{element[0].strip()}:{element[1]}:{element[2]}:{element[3]}\n"
                    file = os.path.abspath(__file__[:-2] + "txt")
                    with open(file, 'w') as f:
                        f.write(text)

                reload.click(fn=reloadPresets, inputs=None, outputs=[lbw_loraratios, lbw_elementals])
                save.click(fn=savePresets, inputs=[lbw_loraratios, lbw_elementals], outputs=None)

        return enabled, lbw_loraratios, lbw_elementals


    def process(self, p, enabled, loraratios, elemental):
        if enabled:
            self.is_flux = getattr(shared.sd_model, 'is_flux', False)
            self.is_zit  = getattr(shared.sd_model, 'is_lumina2', False)

            lratios = {}
            elementals = {}
            for lr in loraratios:
                if lr[0].strip() != "" and len(lr[1].split(",")) in BLOCKNUMS:
                    lratios[lr[0].strip()] = lr[1]
            for el in elemental:
                if el[0].strip() != "":
                    v = ":".join(el[1:])
                    elementals[el[0].strip()] = v

            self.lratios = lratios
            self.elementals = elementals

            on_cfg_denoiser(self.denoiser_callback)

    def denoiser_callback(self, params: CFGDenoiserParams):
        def apply_weight(stop=False):
            if not stop:
                flag_step = self.startsf
            else:
                flag_step = self.stopsf

            lora_patches = shared.sd_model.forge_objects.unet.lora_patches
            refresh_keys = {}
            for m, lf, ef, s, (patch_key, lora_patch) in zip(self.uf, self.lf, self.ef, flag_step, list(lora_patches.items())):
                refresh = False
                for key, vals in lora_patch.items():
                    n_vals = []
                    for v in [v for v in vals if v[1][0] in LORAS]:
                        if s is not None and s == params.sampling_step:
                            if not stop:
                                ratio, _ = ratiodealer(key.replace(".","_"), lf, ef)
                                n_vals.append((ratio * m, *v[1:]))
                            else:
                                n_vals.append((0, *v[1:]))
                            refresh = True
                        else:
                            n_vals.append(v)
                    lora_patch[key] = n_vals
                if refresh:
                    refresh_keys[patch_key] = None

            if len(refresh_keys):
                for refresh_key in list(refresh_keys.keys()):
                    patch = lora_patches[refresh_key]
                    del lora_patches[refresh_key]
                    new_key = (f"{refresh_key[0]}_{str(time.time())}", *refresh_key[1:])
                    refresh_keys[refresh_key] = new_key
                    lora_patches[new_key] = patch

                shared.sd_model.forge_objects.unet.refresh_loras()

                for refresh_key, new_key in list(refresh_keys.items()):
                    patch = lora_patches[new_key]
                    del lora_patches[new_key]
                    lora_patches[refresh_key] = patch

        if params.sampling_step in self.startsf:
            apply_weight()

        if params.sampling_step in self.stopsf:
            apply_weight(stop=True)


    def postprocess(self, p, processed, *args):
        enabled = args[0]
        if enabled:
            lora = importlib.import_module("lora")
            # emb_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()

            # for net in lora.loaded_loras:
                # if hasattr(net,"bundle_embeddings"):
                    # for embedding in net.bundle_embeddings.values():
                        # if embedding.loaded:
                            # emb_db.register_embedding(embedding)

            remove_current_script_callbacks()

            # don't need to clear/unpatch if can reset patches to starting?
            lora.loaded_loras.clear()
            sd_models.model_data.get_sd_model().current_lora_hash = None
            shared.sd_model.forge_objects_after_applying_lora.unet.forge_unpatch_model()
            shared.sd_model.forge_objects_after_applying_lora.clip.patcher.forge_unpatch_model()


    def after_extra_networks_activate(self, p, *args, **kwargs):
        enabled = args[0]
        if enabled:
            loradealer(self, kwargs["prompts"], self.lratios, self.elementals, kwargs["extra_network_data"])


def loradealer(self, prompts, lratios, elementals, extra_network_data = None):
    if extra_network_data is None:
        _, extra_network_data = extra_networks.parse_prompts(prompts)
    moduletypes = extra_network_data.keys()

    for ltype in moduletypes:
        lorars = []
        te_multipliers = []
        unet_multipliers = []
        elements = []
        starts = []
        stops = []
        go_lbw = False
        load = False
        
        if not (ltype == "lora" or ltype == "lyco"):
            continue

        for called in extra_network_data[ltype]:
            items = called.items
            setnow = False
            name = items[0]
            te = syntaxdealer(items, "te=", 1)
            unet = syntaxdealer(items, "unet=", 2)
            te, unet = multidealer(te, unet)

            weights = syntaxdealer(items, "lbw=", 2)
            if weights is None:
                weights = syntaxdealer(items, "w=", 2)
            elem = syntaxdealer(items, "lbwe=", 3)
            start = syntaxdealer(items, "start=", None)
            stop = syntaxdealer(items, "stop=", None)
            steps = syntaxdealer(items, "step=", None)
            if steps is not None:
                start, stop = steps.split("-")
            if start is not None:
                if "." in start:
                    start = int(float(start) * shared.state.sampling_steps)
                else:
                    start = int(start)
                load = True
            if stop is not None:
                if "." in stop:
                    stop = int(float(stop) * shared.state.sampling_steps)
                else:
                    stop = int(stop)
                load = True
            
            if weights is not None and (weights in lratios or any(weights.count(",") == x - 1 for x in BLOCKNUMS)):
                wei = lratios[weights] if weights in lratios else weights
                ratios = [w.strip() for w in wei.split(",")]
                for i, r in enumerate(ratios):
                    if r =="R":
                        ratios[i] = round(random.random(), 3)
                    elif r == "U":
                        ratios[i] = round(random.uniform(-0.5, 1.5), 3)
                    elif r[0] == "X":
                        base = syntaxdealer(items, "x=", 3) if len(items) >= 4 else 1
                        ratios[i] = getinheritedweight(base, r)
                    else:
                        ratios[i] = float(r)
                        
                if len(ratios) not in [26, 30, 61]:
                    ratios = to26(ratios)
                setnow = True
            else:
                if self.is_flux:
                    ratios = [1] * 61
                elif self.is_zit:
                    ratios = [1] * 30
                else:
                    ratios = [1] * 26

            if elem in elementals:
                setnow = True
                elem = elementals[elem]
            else:
                elem = ""

            if setnow or load:
                message = f"[LoRA Block weight] {name}:"
                if setnow:
                    message += f" (Te:{te}, Unet:{unet}) x {ratios}"
                    go_lbw = True
                if load:
                    message += f" (Start:{start}, Stop:{stop})"
                print (message)

            te_multipliers.append(te)
            unet_multipliers.append(unet)
            lorars.append(ratios)
            elements.append(elem)
            starts.append(start)
            stops.append(stop)

        self.startsf = starts
        self.stopsf = stops
        self.uf = unet_multipliers
        self.lf = lorars
        self.ef = elements

        if go_lbw or load:
            lora_patches = shared.sd_model.forge_objects_after_applying_lora.unet.lora_patches 
            lbwf(lora_patches, unet_multipliers, lorars, elements, starts, self.is_flux, self.is_zit)

            lora_patches = shared.sd_model.forge_objects_after_applying_lora.clip.patcher.lora_patches
            lbwf(lora_patches, te_multipliers, lorars, elements, starts, self.is_flux, self.is_zit)


def syntaxdealer(items, target, index): #type "unet=", "x=", "lwbe=" 
    for item in items:
        if target in item:
            return item.replace(target, "")

    if index is None or index + 1 > len(items):
        return None

    if "=" in items[index]:
        return None

    return items[index] if "@" not in items[index] else 1


def multidealer(t, u):
    if t is None and u is None:
        return 1, 1
    elif t is None:
        return float(u), float(u)
    elif u is None:
        return float(t), float(t)
    else:
        return float(t), float(u)


re_inherited_weight = re.compile(r"X([+-])?([\d.]+)?")


def getinheritedweight(weight, offset):
    match = re_inherited_weight.search(offset)
    if match.group(1) == "+":
        return float(weight) + float(match.group(2))
    elif match.group(1) == "-":
        return float(weight) - float(match.group(2))  
    else:
        return float(weight) 


LORAS = ["lora", "loha", "lokr"]

def lbwf(after_applying_lora_patches, ms, lwei, elements, starts, flux, zit):
    errormodules = []
    dict_lora_patches = dict(after_applying_lora_patches.items())

    for m, lw, e, s, hash in zip(ms, lwei, elements, starts, list(after_applying_lora_patches.keys())):
        lora_patches = None
        for k, v in dict_lora_patches.items():
            if k[0] == hash[0]:
                hash = k
                lora_patches = v
                del dict_lora_patches[k]
                break
        if lora_patches is None:
            continue
        for key, vals in lora_patches.items():
            n_vals = []
            lvs = [v for v in vals if v[1][0] in LORAS]
            for v in lvs:
                ratio, picked = ratiodealer(key.replace(".", "_"), lw, e, flux, zit)
                n_vals.append([ratio * m if s is None or s == 0 else 0, *v[1:]])
                if not picked:
                    errormodules.append(key)
            lora_patches[key] = n_vals

        lbw_key = ",".join([str(m)] + [str(int(w) if type(w) is int or w.is_integer() else float(w)) for w in lw]) + e
        new_hash = (hash[0], lbw_key, *hash[2:])

        after_applying_lora_patches[new_hash] = after_applying_lora_patches[hash]
        if new_hash != hash:
            del after_applying_lora_patches[hash]

    if len(errormodules) > 0:
        print("[LoRA Block weight] Unknown modules:", errormodules)


def ratiodealer(key, lwei, elemental:str, flux=False, zit=False):
    ratio = 1
    picked = False
    elemental = elemental.replace("\n", ",")
    elemental = elemental.split(",")
    elemkey = ""
    
    if flux:
        block = elemkey = get_flux_blocks(key)
        if block in BLOCKIDFLUX:
            ratio = lwei[BLOCKIDFLUX.index(block)]
            picked = True
    elif zit:
        block = elemkey = get_zit_blocks(key)
        if block in BLOCKIDZIT:
            ratio = lwei[BLOCKIDZIT.index(block)]
            picked = True
    else:
        for i,block in enumerate(BLOCKS):
            if block in key:
                if i == 26 or i == 27:
                    i = 0
                ratio = lwei[i] 
                picked = True
                elemkey = BLOCKID26[i]

    if len(elemental) > 0:
        skey = key + elemkey
        for d in elemental:
            ds = d.split(":")
            if len(ds) != 3:
                continue
            dbs, dws, dr = (hyphener(ds[0], BLOCKIDFLUX if flux else (BLOCKIDZIT if zit else BLOCKID26)), ds[1], ds[2])
            dbs, dws = (dbs.split(" "), dws.split(" "))
            dbn, dbs = (True, dbs[1:]) if dbs[0] == "NOT" else (False, dbs)
            dwn, dws = (True, dws[1:]) if dws[0] == "NOT" else (False, dws)
            flag = dbn
            for db in dbs:
                if db in skey:
                    flag = not dbn
            if flag:
                flag = dwn
            else:
                continue
            for dw in dws:
                if dw in skey:
                    flag = not dwn
            if flag:
                dr = float(dr)
                ratio = dr
    
    return ratio, picked


LORAANDSOON = {
    "LoraHadaModule" : "w1a",
    "LycoHadaModule" : "w1a",
    "NetworkModuleHada": "w1a",
    "FullModule" : "weight",
    "NetworkModuleFull": "weight",
    "IA3Module" : "w",
    "NetworkModuleIa3" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
    "NetworkModuleLokr": "w1",
    "NetworkModuleGLora": "w1a",
    "NetworkModuleNorm": "w_norm",
    "NetworkModuleOFT": "scale"
}


def hyphener(t, blocks):
    t = t.split(" ")
    for i,e in enumerate(t):
        if "-" in e:
            e = e.split("-")
            if  blocks.index(e[1]) > blocks.index(e[0]):
                t[i] = " ".join(blocks[blocks.index(e[0]):blocks.index(e[1])+1])
            else:
                t[i] = " ".join(blocks[blocks.index(e[1]):blocks.index(e[0])+1])
    return " ".join(t)


def to26(ratios):
    ids = BLOCKIDS[BLOCKNUMS.index(len(ratios))]
    output = [0]*26
    for i, id in enumerate(ids):
        output[BLOCKID26.index(id)] = ratios[i]
    return output


def get_flux_blocks(key):
    if "vae" in key:
        return "VAE"
    if "t5xxl" in key:
        return "T5"
    if "clip_" in key:
        return "CLIP"
    if "t5xxl" in key:
        return "T5"
    
    match = re.search(r'\_(\d+)\_', key)
    if "double_blocks" in key:
        return f"D{match.group(1).zfill(2) }"
    if "single_blocks" in key:
        return f"S{match.group(1).zfill(2) }"
    if "_in" in key:
        return "IN"
    if "final_layer" in key:
        return "OUT"
    return "Not Merge"

def get_zit_blocks(key):
    # if "vae" in key:
        # return "VAE"
    
    match = re.search(r'\_(\d+)\_', key)
    if "diffusion_model_layers" in key:
        return f"J{match.group(1).zfill(2) }"
    return "Not Merge"

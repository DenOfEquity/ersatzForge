import os
# import re
import torch
import network

from backend.args import dynamic_args
from modules import shared, errors, scripts
from backend.utils import load_torch_file
from backend.patcher.lora import model_lora_keys_clip, model_lora_keys_unet, load_lora

import modules_forge.colour_code as cc


def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename="", online_mode=False):
    if model is not None and model.model.diffusion_model.__class__.__name__ == "MiniTrainDIT":
        # Anima LLMAdapter was moved from transformer to text_encoder
        keys = list(lora.keys())
        for k in keys:
            if k.startswith("diffusion_model.llm_adapter"):
                lora[k.replace("diffusion_model.", "qwen3.", 1)] = lora.pop(k)
            elif k.startswith("lora_unet_llm_adapter"):
                lora[k.replace("lora_unet_llm_adapter", "lora_te_llm_adapter", 1)] = lora.pop(k)


        # can't assume start at zero
        def count_blocks(state_dict_keys, prefix):
            split_c = prefix[-1]
            len_prefix = len(prefix)

            max_idx = -1
            for k in state_dict_keys:
                if k.startswith(prefix):
                    bne = k.index(split_c, len_prefix)
                    bn = k[len_prefix:bne]          # lora block number as string

                    idx = int(bn)
                    if idx > max_idx:
                        max_idx = idx

            return max_idx + 1 if max_idx >= 0 else 0


        keys = list(lora.keys())
        if (lora_blocks_count := count_blocks(keys, "lora_unet_blocks_")) > 0:
            prefix = "lora_unet_blocks_"
        else:
            lora_blocks_count = count_blocks(keys, "diffusion_model.blocks.")
            prefix = "diffusion_model.blocks."

        MAPPING = None
        if lora_blocks_count == model.model.diffusion_model.num_blocks:
            pass
        elif 0 < lora_blocks_count <= 28: # lora is probably for standard Anima
            if model.model.diffusion_model.num_blocks == 40: # map to 2.9B
                MAPPING = {
                     "0":[0],        "1":[1, 2],     "2":[3],        "3":[4, 5],
                     "4":[6],        "5":[7, 8],     "6":[9],        "7":[10, 11],
                     "8":[12],       "9":[13, 14],  "10":[15],      "11":[16, 17],
                    "12":[18],      "13":[19],      "14":[20, 21],  "15":[22],
                    "16":[23, 24],  "17":[25],      "18":[26, 27],  "19":[28],
                    "20":[29, 30],  "21":[31],      "22":[32, 33],  "23":[34],
                    "24":[35, 36],  "25":[37],      "26":[38],      "27":[39]
                }
            elif model.model.diffusion_model.num_blocks == 52: # map to 3.8B
                MAPPING = {
                     "0":[0],           "1":[1, 2, 3],     "2":[4],            "3":[5, 6, 7],
                     "4":[8],           "5":[9, 10, 11],   "6":[12],           "7":[13, 14, 15],
                     "8":[16],          "9":[17, 18, 19],  "10":[20],          "11":[21, 22, 23],
                    "12":[24],          "13":[25],         "14":[26, 27, 28],  "15":[29],
                    "16":[30, 31, 32],  "17":[33],         "18":[34, 35, 36],  "19":[37],
                    "20":[38, 39, 40],  "21":[41],         "22":[42, 43, 44],  "23":[45],
                    "24":[46, 47, 48],  "25":[49],         "26":[50],          "27":[51]
                }
        elif lora_blocks_count <= 40: # lora is probably for 2.9B variant
            if model.model.diffusion_model.num_blocks == 28: # map to base
                MAPPING = {
                     "0":[0],   "1":[1],    "2":[],     "3":[2],
                     "4":[3],   "5":[],     "6":[4],    "7":[5],
                     "8":[],    "9":[6],    "10":[7],   "11":[],
                    "12":[8],   "13":[9],   "14":[],    "15":[10],
                    "16":[11],  "17":[],    "18":[12],  "19":[13],
                    "20":[14],  "21":[],    "22":[15],  "23":[16],
                    "24":[],    "25":[17],  "26":[18],  "27":[],
                    "28":[19],  "29":[20],  "30":[],    "31":[21],
                    "32":[22],  "33":[],    "34":[23],  "35":[24],
                    "36":[],    "37":[25],  "38":[26],  "39":[27]
                }
            elif model.model.diffusion_model.num_blocks == 52: # map to 3.8B
                MAPPING = {
                     "0":[0],       "1":[1],        "2":[2, 3],     "3":[4],
                     "4":[5],       "5":[6, 7],     "6":[8],        "7":[9],
                     "8":[10, 11],  "9":[12],       "10":[13],      "11":[14, 15],
                    "12":[16],      "13":[17],      "14":[18, 19],  "15":[20],
                    "16":[21],      "17":[22, 23],  "18":[24],      "19":[25],
                    "20":[26, 27],  "21":[28],      "22":[29],      "23":[30, 31],
                    "24":[32],      "25":[33],      "26":[34, 35],  "27":[36],
                    "28":[37],      "29":[38, 39],  "30":[40],      "31":[41],
                    "32":[42, 43],  "33":[44],      "34":[45],      "35":[46, 47],
                    "36":[48],      "37":[49],      "38":[50],      "39":[51]
                }
        # elif lora_blocks_count <= 52: # lora is for 3.8B variant

        if MAPPING:
            # method 0: adjust block indices with duplication, seems slightly better (tested 28->40)
            split_c = prefix[-1]
            len_prefix = len(prefix)

            new_lora = {}
            for k in list(lora.keys()):
                if k.startswith(prefix):
                    bne = k.index(split_c, len_prefix)
                    bn = k[len_prefix:bne]          # lora block number as string
                    for m in MAPPING[bn]:
                        new_lora[f"{prefix}{m}{k[bne:]}"] = lora[k].clone()
            lora = new_lora

    if model is not None:
        unet_keys = model_lora_keys_unet(model.model)
        lora_unet, lora = load_lora(lora, unet_keys)
    else:
        lora_unet = {}

    if clip is not None:
        clip_keys = model_lora_keys_clip(clip.cond_stage_model) 
        lora_clip, lora = load_lora(lora, clip_keys)
    else:
        lora_clip = {}

    if len(lora) == 0:
        print(f"{cc.LOAD2}[LORA] Loaded {filename}{cc.RESET}")
    else:
        print(f"{cc.LOAD2}[LORA] {cc.WARNING}apparent version mismatch {cc.LOAD2}{filename} {cc.MINOR}ignoring {len(lora)} keys{cc.RESET}")
    del lora

    if len(lora_unet) > 0:
        new_model = model.clone()
        loaded_keys = new_model.add_patches(filename=filename, patches=lora_unet, strength_patch=strength_model, online_mode=online_mode)
        loaded = len(loaded_keys)
        skipped_keys = len(lora_unet) - loaded
        skipped_message = f"; {cc.MINOR}{skipped_keys} keys mismatched{cc.RESET}" if skipped_keys else ""
        print(f"    loaded {loaded} keys for {cc.LOAD2}UNet{cc.RESET} at weight {strength_model} with on_the_fly={online_mode}{skipped_message}")

        if loaded > 0:
            model = new_model

    if len(lora_clip) > 0:
        new_clip = clip.clone()
        loaded_keys = new_clip.add_patches(filename=filename, patches=lora_clip, strength_patch=strength_clip, online_mode=online_mode)
        loaded = len(loaded_keys)
        skipped_keys = len(lora_clip) - loaded
        skipped_message = f"; {cc.MINOR}{skipped_keys} keys mismatched{cc.RESET}" if skipped_keys else ""
        print(f"    loaded {loaded} keys for {cc.LOAD2}CLIP{cc.RESET} at weight {strength_model} with on_the_fly={online_mode}{skipped_message}")

        if loaded > 0:
            clip = new_clip

    return model, clip


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    if shared.sd_model is None:
        return

    loaded_networks = []

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    for i in range(len(names)):
        if networks_on_disk[i] is None:
            print(f"{cc.ERROR}[LoRA] Not found:{cc.RESET} {names[i]}")
            continue
        try:
            net = network.Network(names[i], networks_on_disk[i])
            net.mtime = os.path.getmtime(networks_on_disk[i].filename)
            net.mentioned_name = names[i]
            networks_on_disk[i].read_hash()
            loaded_networks.append(net)
        except Exception as e:
            print(f"{cc.WARNING}[LoRA] {e}{cc.RESET}")
            networks_on_disk[i] = None

    online_mode = dynamic_args.get("online_lora", False)
    if shared.sd_model.forge_objects.unet.model.storage_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        online_mode = False

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        if a is not None:
            compiled_lora_targets.append([a.filename, b, c, online_mode])

    compiled_lora_targets_hash = str(compiled_lora_targets)

    if shared.sd_model.current_lora_hash == compiled_lora_targets_hash:
        return

    shared.sd_model.current_lora_hash = compiled_lora_targets_hash
    shared.sd_model.forge_objects.unet = shared.sd_model.forge_objects_original.unet # cloned, if necessary, in load_lora_for_models()
    shared.sd_model.forge_objects.clip = shared.sd_model.forge_objects_original.clip

    for filename, strength_model, strength_clip, online_mode in compiled_lora_targets:
        lora_sd = load_torch_file(filename, safe_load=True)
        shared.sd_model.forge_objects.unet, shared.sd_model.forge_objects.clip = load_lora_for_models(
            shared.sd_model.forge_objects.unet, shared.sd_model.forge_objects.clip, lora_sd, strength_model, strength_clip,
            filename=filename, online_mode=online_mode)

    shared.sd_model.forge_objects_after_applying_lora = shared.sd_model.forge_objects.shallow_copy()
    return


def process_network_files(names: list[str] | None = None):
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors", ".sft"]))
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


# re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()

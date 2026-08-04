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

import os
import argparse

from backend import utils
from modules.paths_internal import models_path
from modules import shared
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument(
    "--controlnet-dir",
    type=Path,
    help="Path to directory with ControlNet models",
    default=None,
)
parser.add_argument(
    "--controlnet-preprocessor-models-dir",
    type=Path,
    help="Path to directory with annotator model directories",
    default=None,
)

cmd_opts = parser.parse_known_args()[0]

if cmd_opts.controlnet_dir:
    controlnet_dir = str(cmd_opts.controlnet_dir)
else:
    controlnet_dir = os.path.join(models_path, 'ControlNet')
os.makedirs(controlnet_dir, exist_ok=True)

if cmd_opts.controlnet_preprocessor_models_dir:
    preprocessor_dir = str(cmd_opts.controlnet_preprocessor_models_dir)
else:
    preprocessor_dir = os.path.join(models_path, 'ControlNetPreprocessor')
os.makedirs(preprocessor_dir, exist_ok=True)

diffusers_dir = os.path.join(models_path, 'diffusers')
os.makedirs(diffusers_dir, exist_ok=True)

supported_preprocessors = {}
supported_control_models = []


def add_supported_preprocessor(preprocessor):
    global supported_preprocessors
    p = preprocessor
    supported_preprocessors[p.name] = p
    return


def add_supported_control_model(control_model):
    global supported_control_models
    supported_control_models.append(control_model)
    return


# caching by Haoming02, ForgeNeo
from collections import OrderedDict

class ModelCache:
    def __init__(self):
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, model):
        self.cache[key] = model
        self.cache.move_to_end(key)

        max_size = max(1, shared.opts.data.get("control_net_model_cache_size", 3))
        while len(self.cache) > max_size:
            self.cache.popitem(last=False)

_CONTROL_MODEL_CACHE = ModelCache()


def try_load_supported_control_model(ckpt_path):
    global supported_control_models
    global _CONTROL_MODEL_CACHE

    control_model = _CONTROL_MODEL_CACHE.get(ckpt_path)

    if control_model is None:
        state_dict = utils.load_torch_file(ckpt_path, safe_load=True)
        for supported_type in supported_control_models:
            state_dict_copy = dict(state_dict)
            control_model = supported_type.try_build_from_state_dict(state_dict_copy, ckpt_path)
            if control_model is not None:
                _CONTROL_MODEL_CACHE.put(ckpt_path, control_model)
                break

    return control_model

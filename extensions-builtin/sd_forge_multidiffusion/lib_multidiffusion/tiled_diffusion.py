# 1st Edit by. https://github.com/shiimizu/ComfyUI-TiledDiffusion
# 2nd Edit by. Forge Official
# 3rd Edit by. Haoming02
# - Based on: https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111

# dead code removed


from typing import Final, Optional, Union

import numpy
import torch
from torch import Tensor

from backend import memory_management
from backend.misc.image_resize import adaptive_resize
from backend.patcher.base import ModelPatcher
from backend.patcher.controlnet import ControlNet, T2IAdapter

_dev: Final[torch.device] = memory_management.get_torch_device()

opt_f: Optional[int] = None


class BBox:

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x + w, y + h]
        self.slicer = slice(None), slice(None), slice(y, y + h), slice(x, x + w)

    def __getitem__(self, idx: int) -> int:
        return self.box[idx]


def ceildiv(big, small):
    return -(big // -small)


def split_bboxes(w: int, h: int, tile_w: int, tile_h: int, overlap: int = 16, init_weight: Union[Tensor, float] = 1.0) -> tuple[list[BBox], Tensor]:
    cols = ceildiv((w - overlap), (tile_w - overlap))
    rows = ceildiv((h - overlap), (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: list[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=_dev, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight


class AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__

        self.w: int = 0
        self.h: int = 0
        self.tile_width: int = None
        self.tile_height: int = None
        self.tile_overlap: int = None
        self.tile_batch_size: int = None
        self.step_count = 0
        self.inner_loop_count = 0
        self.num_tiles: int = None
        self.num_batches: int = None
        self.batched_bboxes: list[list[BBox]] = []
        self.enable_controlnet: bool = False
        self.control_params: dict[tuple, list[list[Tensor]]] = {}

        self.weights = None

    def repeat_tensor(self, x: Tensor, n: int, concat=False, concat_to=0) -> Tensor:
        if n == 1:
            return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:
            shape = [n] + [-1] * r_dims
            return x.expand(shape)
        else:
            if concat:
                return torch.cat([x for _ in range(n)], dim=0)[:concat_to]
            shape = [n] + [1] * r_dims
            return x.repeat(shape)

    def init_grid_bbox(self):
        self.weights = torch.zeros((1, 1, self.h, self.w), device=_dev, dtype=torch.float32)

        overlap = max(0, min(self.tile_overlap, min(self.tile_width, self.tile_height) - 4))
        self.tile_width = min(self.tile_width, self.w)
        self.tile_height = min(self.tile_height, self.h)
        bboxes, weights = split_bboxes(self.w, self.h, self.tile_width, self.tile_height, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_tiles = len(bboxes)
        self.num_batches = ceildiv(self.num_tiles, self.tile_batch_size)
        self.tile_batch_size = ceildiv(len(bboxes), self.num_batches)
        self.batched_bboxes = [bboxes[i * self.tile_batch_size : (i + 1) * self.tile_batch_size] for i in range(self.num_batches)]

    def get_tile_weights(self) -> Union[Tensor, float]:
        return 1.0

    def process_controlnet(self, x_shape, x_dtype, c_in: dict, cond_or_uncond: list, bboxes, batch_size: int, batch_id: int):
        control: ControlNet = c_in["control_model"]
        param_id = -1
        tuple_key = tuple(cond_or_uncond) + tuple(x_shape)
        while control is not None:
            param_id += 1
            PH, PW = self.h * 8, self.w * 8

            if self.control_params.get(tuple_key, None) is None:
                self.control_params[tuple_key] = [[None]]
                val = self.control_params[tuple_key]
                if param_id + 1 >= len(val):
                    val.extend([[None] for _ in range(param_id + 1)])
                if len(self.batched_bboxes) >= len(val[param_id]):
                    val[param_id].extend([[None] for _ in range(len(self.batched_bboxes))])
            if self.refresh or control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], Tensor):
                dtype = getattr(control, "manual_cast_dtype", None)
                if dtype is None:
                    dtype = getattr(getattr(control, "control_model", None), "dtype", None)
                if dtype is None:
                    dtype = x_dtype
                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    control.cond_hint = adaptive_resize(control.cond_hint_original, width, height, "nearest-exact", "center").float().to(control.device)
                    if control.channels_in == 1 and control.cond_hint.shape[1] > 1:
                        control.cond_hint = torch.mean(control.cond_hint, 1, keepdim=True)
                else:
                    if (PH, PW) == (control.cond_hint_original.shape[-2], control.cond_hint_original.shape[-1]):
                        control.cond_hint = control.cond_hint_original.clone().to(dtype=dtype, device=control.device)
                    else:
                        control.cond_hint = adaptive_resize(control.cond_hint_original, PW, PH, "nearest-exact", "center").to(dtype=dtype, device=control.device)
                cond_hint_pre_tile = control.cond_hint
                if control.cond_hint.shape[0] < batch_size:
                    cond_hint_pre_tile = self.repeat_tensor(control.cond_hint, ceildiv(batch_size, control.cond_hint.shape[0]))[:batch_size]
                cns = [cond_hint_pre_tile[:, :, bbox[1] * opt_f : bbox[3] * opt_f, bbox[0] * opt_f : bbox[2] * opt_f] for bbox in bboxes]
                control.cond_hint = torch.cat(cns, dim=0)
                self.control_params[tuple_key][param_id][batch_id] = control.cond_hint
            else:
                control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet


class MultiDiffusion(AbstractDiffusion):

    @torch.no_grad()
    def __call__(self, model_function, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: list = args["cond_or_uncond"]
        c_crossattn: Tensor = c_in["c_crossattn"]

        N, C, H, W = x_in.shape
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox()
        self.h, self.w = H, W
        
        x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)

        for batch_id, bboxes in enumerate(self.batched_bboxes):
            x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
            n_rep = len(bboxes)
            ts_tile = self.repeat_tensor(t_in, n_rep)
            cond_tile = self.repeat_tensor(c_crossattn, n_rep)
            c_tile = c_in.copy()
            c_tile["c_crossattn"] = cond_tile
            if "time_context" in c_in:
                c_tile["time_context"] = self.repeat_tensor(c_in["time_context"], n_rep)
            for key in c_tile:
                if key in ["y", "c_concat"]:
                    icond = c_tile[key]

                    if icond.shape[2:] == (self.h, self.w):
                        c_tile[key] = torch.cat([icond[bbox.slicer] for bbox in bboxes])
                    else:
                        c_tile[key] = self.repeat_tensor(icond, n_rep)
            if "control" in c_in:
                self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, bboxes, N, batch_id)
                c_tile["control"] = c_in["control_model"].get_control(x_tile, ts_tile, c_tile, len(cond_or_uncond))

            x_tile_out = model_function(x_tile, ts_tile, **c_tile)

            for i, bbox in enumerate(bboxes):
                x_buffer[bbox.slicer] += x_tile_out[i * N : (i + 1) * N, :, :, :]
            del x_tile_out, x_tile, ts_tile, c_tile
        x_out = torch.where(self.weights > 1, x_buffer / self.weights, x_buffer)

        return x_out


class MixtureOfDiffusers(AbstractDiffusion):
    """
    Mixture-of-Diffusers Implementation
    https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights: list[Tensor] = []

    def get_tile_weights(self) -> Tensor:
        def f(x, midpoint, var=0.01):
            return numpy.exp(-(x - midpoint) * (x - midpoint) / (self.tile_width * self.tile_width) / (2 * var)) / numpy.sqrt(2 * numpy.pi * var)
        x_probs = [f(x, (self.tile_width - 1) / 2) for x in range(self.tile_width)]
        y_probs = [f(y, self.tile_height / 2) for y in range(self.tile_height)]

        w = numpy.outer(y_probs, x_probs)
        self.tile_weights = torch.from_numpy(w).to(_dev, dtype=torch.float32)

        return self.tile_weights

    @torch.no_grad()
    def __call__(self, model_function, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: list = args["cond_or_uncond"]
        c_crossattn: Tensor = c_in["c_crossattn"]

        N, C, H, W = x_in.shape

        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox()
            self.rescale_factor = 1 / self.weights
        self.h, self.w = H, W

        x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)

        for batch_id, bboxes in enumerate(self.batched_bboxes):
            x_tile_list = []
            t_tile_list = []
            icond_map = {}
            for bbox in bboxes:
                x_tile_list.append(x_in[bbox.slicer])
                t_tile_list.append(t_in)
                if isinstance(c_in, dict):
                    for key in ["y", "c_concat"]:
                        if key in c_in:
                            icond = c_in[key]

                            if icond.shape[2:] == (self.h, self.w):
                                icond = icond[bbox.slicer]
                            if icond_map.get(key, None) is None:
                                icond_map[key] = []
                            icond_map[key].append(icond)
                else:
                    print(">> [WARN] not supported, make an issue on github!!")
            n_rep = len(bboxes)
            x_tile = torch.cat(x_tile_list, dim=0)
            t_tile = self.repeat_tensor(t_in, n_rep)
            tcond_tile = self.repeat_tensor(c_crossattn, n_rep)
            c_tile = c_in.copy()
            c_tile["c_crossattn"] = tcond_tile
            if "time_context" in c_in:
                c_tile["time_context"] = self.repeat_tensor(c_in["time_context"], n_rep)
            for key in c_tile:
                if key in ["y", "c_concat"]:
                    icond_tile = torch.cat(icond_map[key], dim=0)
                    c_tile[key] = icond_tile
            if "control" in c_in:
                self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, bboxes, N, batch_id)
                c_tile["control"] = c_in["control_model"].get_control(x_tile, t_tile, c_tile, len(cond_or_uncond))

            x_tile_out = model_function(x_tile, t_tile, **c_tile)

            for i, bbox in enumerate(bboxes):
                w = self.tile_weights * self.rescale_factor[bbox.slicer]
                x_buffer[bbox.slicer] += x_tile_out[i * N : (i + 1) * N, :, :, :] * w
            del x_tile_out, x_tile, t_tile, c_tile
        x_out = x_buffer

        return x_out


class TiledDiffusion:

    @staticmethod
    def apply(model: ModelPatcher, method: str, tile_width: int, tile_height: int, tile_overlap: int, tile_batch_size: int):
        match method:
            case "MultiDiffusion":
                implement = MultiDiffusion()
            case "Mixture of Diffusers":
                implement = MixtureOfDiffusers()
            case _:
                raise SystemError

        from modules import processing

        global opt_f
        opt_f = processing.opt_f

        implement.tile_width = tile_width // opt_f
        implement.tile_height = tile_height // opt_f
        implement.tile_overlap = tile_overlap // opt_f
        implement.tile_batch_size = tile_batch_size

        model = model.clone()
        model.set_model_unet_function_wrapper(implement)
        model.model_options["tiled_diffusion"] = True
        return model

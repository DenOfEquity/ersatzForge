import contextlib
import torch
from backend import memory_management


def has_xpu() -> bool:
    return memory_management.xpu_available


def has_mps() -> bool:
    return memory_management.mps_mode()


def cuda_no_autocast(device_id=None) -> bool:
    return False


def get_cuda_device_id():
    return memory_management.get_torch_device().index


def get_cuda_device_string():
    return str(memory_management.get_torch_device())


def get_optimal_device_name():
    return memory_management.get_torch_device().type


def get_optimal_device():
    return memory_management.get_torch_device()


def get_device_for(task):
    return memory_management.get_torch_device()


def torch_gc():
    memory_management.soft_empty_cache()


def torch_npu_set_device():
    return


def enable_tf32():
    return


cpu: torch.device = torch.device("cpu")
fp8: bool = False
device: torch.device = memory_management.get_torch_device()
device_face_restore: torch.device = memory_management.get_torch_device()  # will be managed by memory management system, used for GFPGAN and codeformer
dtype: torch.dtype = torch.float32 if memory_management.unet_dtype() is torch.float32 else torch.float16
dtype_vae: torch.dtype = memory_management.vae_dtype()
dtype_unet: torch.dtype = memory_management.unet_dtype()
dtype_inference: torch.dtype = memory_management.unet_dtype()


def cond_cast_unet(input):
    return input


def cond_cast_float(input):
    return input


nv_rng = None
patch_module_list = []


def manual_cast_forward(target_dtype):
    return


@contextlib.contextmanager
def manual_cast(target_dtype):
    return


def autocast(disable=False):
    return contextlib.nullcontext()


def without_autocast(disable=False):
    return contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    return


from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import HWC3
from modules import devices
from modules.modelloader import load_file_from_url
from backend.memory_management import should_use_fp16

import os
import torch
import numpy
import cv2

from torchvision.transforms import Compose
from safetensors.torch import load_file

from depth_anything.dpt import DPT_DINOv2
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class PreprocessorDepthAnything(Preprocessor):
    """https://github.com/LiheYoung/Depth-Anything"""
    """https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2"""
    """https://github.com/ByteDance-Seed/Depth-Anything-3"""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ["Depth"]
        self.model_filename_filters = ["depth"]
        self.slider_resolution = PreprocessorParameter(label="Resolution", minimum=140, maximum=2072, value=518, step=14, visible=True)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.model = None
        self.device = devices.get_device_for("controlnet")
        self.dtype = torch.float16 if should_use_fp16(self.device, prioritize_performance=False, manual_cast=True) else torch.float32

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        model_dir = os.path.join(preprocessor_dir, "depth_anything")
        remote_model_path = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
        model_path = os.path.join(model_dir, "depth_anything_vitl14.pth")
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = DPT_DINOv2(
                    encoder="vitl",
                    features=256,
                    out_channels=[256, 512, 1024, 1024],
                    localhub=False,
                )

        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        self.model = model.to(self.dtype)

    def load_model_v2(self):
        model_dir = os.path.join(preprocessor_dir, "depth_anything_v2")
        remote_model_path = "https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors"
        model_path = os.path.join(model_dir, 'depth_anything_v2_vitl.safetensors')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = DepthAnythingV2(
                    encoder="vitl",
                    features=256,
                    out_channels=[256, 512, 1024, 1024],
                )

        model.load_state_dict(load_file(model_path))
        model.eval()
        self.model = model.to(self.dtype)

    def load_model_v3(self):
        from depth_anything_3.api import DepthAnything3

        # model = DepthAnything3.from_pretrained("depth-anything/da3metric-large", local_files_only=True)
        model = DepthAnything3.from_pretrained("depth-anything/da3mono-large", local_files_only=True)
        self.model = model.to(self.dtype)

    def __call__(self, input_image, resolution=518, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.model is None:
            if self.name == "depth_anything_v3":
                self.load_model_v3()
            elif self.name == "depth_anything_v2":
                self.load_model_v2()
            else:
                self.load_model()
        self.model.to(self.device)

        transform = Compose(
            [
                Resize(
                    width=resolution,
                    height=resolution,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        image = input_image / 255.0 #cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(self.dtype)

        with torch.no_grad():
            if self.name == "depth_anything_v3":
                depth = self.model.inference(image).depth
                # depth /= (1.0 + depth)
            else:
                depth = self.model(image.to(self.device))
            depth -= depth.min()
            depth /= depth.max()
            depth **= 0.45455
            if self.name == "depth_anything_v3":
                depth = 1.0 - depth
            depth *= 255.0
            if self.name == "depth_anything_v3":
                result = depth.astype(numpy.uint8)
            else:
                result = depth.cpu().numpy().astype(numpy.uint8)

        self.model.cpu()
        torch.cuda.empty_cache()

        result = numpy.moveaxis(result, 0, -1)
        return HWC3(result)


add_supported_preprocessor(PreprocessorDepthAnything("depth_anything"))
add_supported_preprocessor(PreprocessorDepthAnything("depth_anything_v2"))
add_supported_preprocessor(PreprocessorDepthAnything("depth_anything_v3"))

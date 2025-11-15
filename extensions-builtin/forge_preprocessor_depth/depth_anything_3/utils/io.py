# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input processor for Depth Anything 3 (parallelized).
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from depth_anything_3.specs import Prediction

# class InputProcessor:
    # NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # def __init__(self):
        # pass

    # def __call__(
        # self,
        # image: list[np.ndarray | Image.Image | str],
        # process_res: int = 504,
    # ) -> tuple[torch.Tensor, list[np.ndarray | None], list[np.ndarray | None]]:

        # pil_img = Image.fromarray(image[0]).convert("RGB")

        # pil_img = self._resize_shortest_side(pil_img, process_res)
        # w, h = pil_img.size

        # img_tensor = self._normalize_image(pil_img)
        # _, H, W = img_tensor.shape
        # assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

        # return img_tensor.unsqueeze(0)


    # def _normalize_image(self, img: Image.Image) -> torch.Tensor:
        # img_tensor = T.ToTensor()(img)
        # return self.NORMALIZE(img_tensor)


    # def _resize_shortest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        # w, h = img.size

        # shortest = min(w, h)

        # scale = target_size / float(shortest)
        # new_w = max(1, int(round(w * scale))) + (14 // 2)
        # new_h = max(1, int(round(h * scale))) + (14 // 2)

        # new_w = 14 * (new_w // 14)
        # new_h = 14 * (new_h // 14)

        # if new_w == w and new_h == h:
            # return img

        # upscale = (new_w > w) or (new_h > h)
        # interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA

        # arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        # return Image.fromarray(arr)


class OutputProcessor:
    def __init__(self) -> None:
        """Initialize the output processor."""

    def __call__(self, model_output: dict[str, torch.Tensor]) -> Prediction:
        depth = model_output["depth"].squeeze(0).squeeze(-1).cpu().numpy()  # (N, H, W) self._extract_depth(model_output)
        # sky = model_output.get("sky", None)
        # if sky is not None:
            # sky = sky.squeeze(0).cpu().numpy() >= 0.5  # (N, H, W)
        return Prediction(
            depth=depth,
            # sky=sky,
            # is_metric=getattr(model_output, "is_metric", 0),
        )

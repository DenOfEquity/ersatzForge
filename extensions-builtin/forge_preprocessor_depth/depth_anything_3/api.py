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
Depth Anything 3 API module.

This module provides the main API for Depth Anything 3, including model loading,
inference, and export capabilities. It supports both single and nested model architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from omegaconf import OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction

torch.backends.cudnn.benchmark = False


SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.

    This class provides a high-level interface for depth estimation using Depth Anything 3.
    It supports both single and nested model architectures with metric scaling capabilities.

    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Automatic mixed precision inference
    - Export capabilities for various formats (GLB, PLY, NPZ, etc.)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")

        # Run inference
        prediction = model.inference(images, export_dir="output", export_format="glb")
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub


    def __init__(self, model_name: str = "da3-large", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name

        # Build the underlying network
        self.config = OmegaConf.load(MODEL_REGISTRY[self.model_name]) #load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

        # Device management (set by user)
        self.device = None

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.

        Returns:
            Dictionary containing model predictions
        """
        with torch.no_grad():
            return self.model(image, None, None, [], False)


    def inference(
        self,
        image,
    ) -> Prediction:

        # Move images to model device
        device = self._get_model_device()
        imgs = image.to(device, non_blocking=True)[None]

        # Run model forward pass
        raw_output = self._run_model_forward(imgs)

        depth = raw_output["depth"].squeeze(0).squeeze(-1).detach() # (N, H, W)
        sky = raw_output["sky"].squeeze(0).detach() # (N, H, W)
        return Prediction(depth=depth, sky=sky)


    def _run_model_forward(
        self,
        imgs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        output = self.forward(imgs)
        if need_sync:
            torch.cuda.synchronize(device)
        return output


    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")

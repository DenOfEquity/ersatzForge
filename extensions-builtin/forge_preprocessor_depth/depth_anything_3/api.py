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

from typing import Sequence
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
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.

        Returns:
            Dictionary containing model predictions
        """
        with torch.no_grad():
            return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs)

    def inference(
        self,
        image,
        export_feat_layers: Sequence[int] | None = None,
    ) -> Prediction:

        # Prepare tensors for model
        device = self._get_model_device()

        # Move images to model device
        imgs = image.to(device, non_blocking=True)[None]

        # Run model forward pass
        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self._run_model_forward(imgs, export_feat_layers)

        depth = raw_output["depth"].squeeze(0).squeeze(-1).cpu().numpy()  # (N, H, W) self._extract_depth(model_output)
        # sky = raw_output.get("sky", None)
        # if sky is not None:
            # sky = sky.squeeze(0).cpu().numpy() >= 0.5  # (N, H, W)
        return Prediction(
            depth=depth,
            # sky=sky,
            # is_metric=getattr(raw_output, "is_metric", 0),
        )


    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        export_feat_layers: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output = self.forward(imgs, None, None, feat_layers, False)
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

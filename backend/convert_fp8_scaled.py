import os
import re
import torch

from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm

# original: Clybius, https://github.com/Clybius/Learned-Rounding

# Keys containing these strings will not be quantized if a given argument is set
AVOID_KEY_NAMES = []#"norm", "bias", "embed_tokens", "shared"] #T5XXL, may need to be changed for other TEs.
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32 # Don't think more hurts here since we're working tensor by tensor.
# Dtype for storing scale factors
SCALE_DTYPE = torch.float32

class LearnedRoundingConverter:
    """
    Implements adaptive rounding for converting a weight to float8.
    Inspired by AdaRound paper (https://arxiv.org/abs/2004.10568).
    "TPEC-Quant" (Top-Principal Error Correction Quantization)
    """
    def __init__(self, num_iter=256, top_k=1):
        self.num_iter = num_iter
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # The maximum representable value for e4m3fn, used for scaling.
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"LearnedRoundingConverter initialized on device: {self.device}")

    def convert(self, W_orig: torch.Tensor, verbose=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#        Performs the learned rounding conversion for a single weight tensor.

        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        # Step 1: Calculate the quantization scale (per-tensor asymmetric)
        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            if verbose:
                print("  - Tensor is all zeros, skipping optimization.")
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), None, torch.zeros_like(W_float32).cpu()

        scale = self.f8_max_val / w_max # Example: (absmax = 1, fp8 max = +-448 for dtype e4m3_fn)
        W_scaled = W_float32 * scale # absmax now +-448

        # Step 2: Initialize the rounding mask 'h'
        W_rounded = W_scaled.to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE) # Naive RtN quantization on scaled model
        k = min(self.top_k, min(W_float32.shape))
        U, _, Vh = torch.pca_lowrank(W_float32, q=k, center=False, niter=8) # To my knowledge, LAPACK (or magma or w/e) uses 1k iters by default. Unsure if the default of 2 is good so set it to 16 here.
        Vh = Vh.T
        U_k = U[:, :k] # Obtain most important low-rank matrices
        Vh_k = Vh[:k, :]

        W_q_refined = W_rounded.clone() # Clone, as this tensor will be the one thats iteratively refined

        # Step 4: The optimization loop
        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        lr = 1.0
        curr_lr = lr
        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False, disable=not verbose)
        for i in pbar:
            current_dq = W_q_refined / scale
            error = current_dq - W_float32

            projected_error = U_k.T @ error @ Vh_k.T

            loss = torch.linalg.norm(projected_error)**2

            if loss.abs() < 1e-8:
                if verbose:
                    print(f"Loss {loss.item():.9f} is negligible. Stopping at iteration {i}.")
                break
            
            # Simple learning rate scheduler and early stopping
            if loss.abs() >= best_loss:
                worse_loss_counter += 1
                curr_lr = max(curr_lr / 2, 1e-8)
                if worse_loss_counter >= 40: # Reduce LR after 20 worse iterations
                    if verbose:
                        print(f"Loss ({best_loss}) has only gotten worse over {worse_loss_counter} iterations, keeping best tensor and skipping...")
                    break
            else:
                best_loss = loss.abs().item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = curr_lr * 2
            
            grad = U_k @ projected_error @ Vh_k

            W_q_refined = W_q_refined - curr_lr * grad

            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

        final_tensor = best_tensor if best_tensor is not None else W_q_refined

        # Final Hard Quantization
        with torch.no_grad():
            W_f8 = final_tensor.to(TARGET_FP8_DTYPE)

        # Calculate dequantization scale (reciprocal of the quantization scale)
        dequant_scale = scale.reciprocal().reshape(1)
        # Clean up GPU memory
        del W_float32, W_scaled, W_rounded, W_q_refined, error, U, Vh, U_k, Vh_k

        return W_f8.cpu(), dequant_scale.cpu(), (W_f8.to(COMPUTE_DTYPE) * dequant_scale).cpu()


    def convert_row_fp8(self, W_orig: torch.Tensor, verbose=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        row_max_abs = float(self.f8_max_val) / W_float32.abs().amax(dim=1)

        with torch.no_grad():
            scales_per_row = torch.where(row_max_abs == 0, 
                                         torch.ones_like(row_max_abs), 
                                         row_max_abs).unsqueeze(1)

        W_scaled = W_float32 * scales_per_row # absmax now +-448

        W_rounded = W_scaled.to(TARGET_FP8_DTYPE)

        dequant_scale = scales_per_row.reciprocal()

        del W_float32, W_scaled

        return W_rounded.cpu(), dequant_scale.cpu(), (W_rounded.to(COMPUTE_DTYPE) * dequant_scale).cpu()


    def convert_row_int8(self, W_orig: torch.Tensor, verbose=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        row_max_abs = 127.0 / W_float32.abs().amax(dim=1)

        with torch.no_grad():
            scales_per_row = torch.where(row_max_abs == 0, 
                                         torch.ones_like(row_max_abs), 
                                         row_max_abs).unsqueeze(1)

        W_scaled = W_float32 * scales_per_row

        W_rounded = W_scaled.round().to(torch.int8)

        dequant_scale = scales_per_row.reciprocal()

        del W_float32, W_scaled

        return W_rounded.cpu(), dequant_scale.cpu(), (W_rounded.to(COMPUTE_DTYPE) * dequant_scale).cpu()


    def convert_row_mixed(self, W_orig: torch.Tensor, verbose=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        row_max_int8 = 127.0 / W_float32.abs().amax(dim=1)
        row_max_fp8 = 448.0 / W_float32.abs().amax(dim=1)

        with torch.no_grad():
            scales_per_row_int8 = torch.where(row_max_int8 == 0, 
                                         torch.ones_like(row_max_int8), 
                                         row_max_int8).unsqueeze(1)
            scales_per_row_fp8 = torch.where(row_max_fp8 == 0, 
                                         torch.ones_like(row_max_fp8), 
                                         row_max_fp8).unsqueeze(1)

        W_scaled_int8 = W_float32 * scales_per_row_int8
        W_rounded_int8 = W_scaled_int8.round().to(torch.int8)
        dequant_scale_int8 = scales_per_row_int8.reciprocal()
        dequant_int8 = W_rounded_int8.to(torch.float32) * dequant_scale_int8
        similarity_int8 = 1.0 - torch.nn.functional.cosine_similarity(
                dequant_int8.flatten().unsqueeze(0),  # shape (1, N)
                W_float32.flatten().unsqueeze(0),  # shape (1, N)
                dim=1,
                eps=1e-8
            ).item()
        mse_int8 = torch.nn.functional.mse_loss(dequant_int8.flatten(), W_float32.flatten()).item()
        distance_int8 = similarity_int8 + mse_int8

        W_scaled_fp8 = W_float32 * scales_per_row_fp8
        W_rounded_fp8 = W_scaled_fp8.to(torch.float8_e4m3fn)
        dequant_scale_fp8 = scales_per_row_fp8.reciprocal()
        dequant_fp8 = W_rounded_fp8.to(torch.float32) * dequant_scale_fp8
        similarity_fp8 = 1.0 - torch.nn.functional.cosine_similarity(
                dequant_fp8.flatten().unsqueeze(0),  # shape (1, N)
                W_float32.flatten().unsqueeze(0),  # shape (1, N)
                dim=1,
                eps=1e-8
            ).item()
        mse_fp8 = torch.nn.functional.mse_loss(dequant_fp8.flatten(), W_float32.flatten()).item()
        distance_fp8 = similarity_fp8 + mse_fp8

        del W_float32

        if distance_fp8 < distance_int8:
            return W_rounded_fp8.cpu(), dequant_scale_fp8.cpu(), (W_rounded_fp8.to(COMPUTE_DTYPE) * dequant_scale_fp8).cpu()
        else:
            return W_rounded_int8.cpu(), dequant_scale_int8.cpu(), (W_rounded_int8.to(COMPUTE_DTYPE) * dequant_scale_int8).cpu()


def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

# Global FP8 constants
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

import backend.utils as utils


def convert_to_fp8_scaled(input_file: str, avoid_keys: str, remove_keys: str, mode: str, output_file: str, t5xxl: bool, keep_distillation: bool, calib_samples: int, iterations: int, top_k: int, verbose: bool, row: bool=False):
    """
    Converts a safetensors file to a version with scaled weights (per row or using learned rounding (modified from AdaRound)).
    """
    print(f"Processing: {input_file}")

    tensors: Dict[str, torch.Tensor] = {}
    
    input_ext = os.path.splitext(input_file)[1]

    tensors = utils.load_torch_file(input_file, safe_load=True)

    if any(k.endswith(".scale_weight") for k in tensors.keys()):
        print("Model is already converted!")
        return "Model is already converted!"

    key_prefix = os.path.commonprefix(list(tensors.keys()))

    print(f"Output will be saved to: {output_file}")

    if remove_keys != "":
        remove = re.compile(remove_keys)
        for key in list(tensors):
            if re.search(remove, key):
                tensors.pop(key)

    if avoid_keys != "":
        avoid = re.compile(avoid_keys)
    else:
        avoid = None

    # Instantiate the converter with hyperparameters from command line
    converter = LearnedRoundingConverter(iterations, top_k)

    # generate calibration data for each unique input dimension as each dimension is found
    calibration_data_cache = {}

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith(".weight")])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")

    if mode not in ["fp8_scaled", "int8", "mixed8"]:  # no scaling
        for i, key in enumerate(weight_keys):
            process_this_key = True

            if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
                if verbose:
                    print(f"({i+1}/{total_weights}) Skipping excluded T5XXL tensor: {key}")
                new_tensors[key] = tensors[key]
                process_this_key = False
                skipped_count += 1

            if avoid is not None:
                if re.search(avoid, key):
                    if verbose:
                        print(f"({i+1}/{total_weights}) Skipping due to avoid list match: {key}")
                    new_tensors[key] = tensors[key]
                    process_this_key = False
                    skipped_count += 1

            if keep_distillation and any(avoid_name in key for avoid_name in DISTILL_LAYER_KEYNAMES):
                if verbose:
                    print(f"({i+1}/{total_weights}) Skipping excluded distillation tensor: {key}")
                new_tensors[key] = tensors[key]
                process_this_key = False
                skipped_count += 1

            if not process_this_key:
                continue
            processed_count += 1

            if mode == "fp16":
                new_tensors[key] = tensors[key].to(torch.float16)
            else:
                new_tensors[key] = tensors[key].to(TARGET_FP8_DTYPE)

        for key, tensor in tensors.items():
            if key not in new_tensors:
                new_tensors[key] = tensor
                if verbose:
                    print(f"(+) Adding original non-quantized tensor: {key}")

    else:
        for i, key in enumerate(weight_keys):
            process_this_key = True

            if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
                if verbose:
                    print(f"({i+1}/{total_weights}) Skipping excluded T5XXL tensor: {key}")
                new_tensors[key] = tensors[key]
                process_this_key = False
                skipped_count += 1

            if avoid is not None:
                if re.search(avoid, key):
                    if verbose:
                        print(f"({i+1}/{total_weights}) Skipping due to avoid list match: {key}")
                    new_tensors[key] = tensors[key]
                    process_this_key = False
                    skipped_count += 1

            if keep_distillation and any(avoid_name in key for avoid_name in DISTILL_LAYER_KEYNAMES):
                if verbose:
                    print(f"({i+1}/{total_weights}) Skipping excluded distillation tensor: {key}")
                new_tensors[key] = tensors[key]
                process_this_key = False
                skipped_count += 1

            if not process_this_key:
                continue

            if verbose:
                print(f"({i+1}/{total_weights}) Processing tensor: {key}")
            else:
                print(f"({i+1}/{total_weights})", end="\r")
            processed_count += 1

            original_tensor = tensors[key]
            base_name = key[:-len(".weight")]

            if original_tensor.numel() == 0 or original_tensor.ndim < 2:
                if verbose:
                    print(f"  - Dumb convert of empty or non-2D tensor: {key}")
                quantized_fp8_tensor = tensors[key].to(TARGET_FP8_DTYPE)
                dequantized_weight_tensor = quantized_fp8_tensor.to(COMPUTE_DTYPE)
                dequant_scale = None
                new_tensors[key] = quantized_fp8_tensor
            else:
                if mode == "mixed8":
                    quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert_row_mixed(original_tensor.flatten(1), verbose)
                elif mode == "int8":
                    quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert_row_int8(original_tensor.flatten(1), verbose)
                elif row:
                    quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert_row_fp8(original_tensor.flatten(1), verbose)
                else:
                    quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert(original_tensor.flatten(1), verbose)

                quantized_fp8_tensor = quantized_fp8_tensor.reshape(original_tensor.shape)
                dequantized_weight_tensor = dequantized_weight_tensor.reshape(original_tensor.shape)
                dequant_scale = dequant_scale.reshape([original_tensor.shape[0]] + [1] * (original_tensor.ndim-1))

                # Store the results
                new_tensors[key] = quantized_fp8_tensor
                if dequant_scale is not None:
                    scale_weight_key = f"{base_name}.scale_weight"
                    new_tensors[scale_weight_key] = dequant_scale.to(SCALE_DTYPE)

            # --- BIAS CORRECTION ---
            bias_key = f"{base_name}.bias"
            if original_tensor.ndim >= 2 and bias_key in tensors:
                in_features = original_tensor.flatten(1).shape[1]
                if in_features not in calibration_data_cache:
                    if verbose:
                        print(f"  - Found new in_features dimension: {in_features}. Generating calibration data.")
                    calibration_data_cache[in_features] = torch.randn(
                        calib_samples, in_features, dtype=COMPUTE_DTYPE
                    )

                calibration_data = calibration_data_cache[in_features]

                if verbose:
                    print(f"  - Found and adjusting corresponding bias: {bias_key}")
                with torch.no_grad():
                    original_bias = tensors[bias_key]
                    
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # Move tensors to the compute device
                    W_orig_dev = original_tensor.flatten(1).to(device, dtype=COMPUTE_DTYPE)
                    W_dequant_dev = dequantized_weight_tensor.flatten(1).to(device, dtype=COMPUTE_DTYPE)
                    X_calib_dev = calibration_data.to(device, dtype=COMPUTE_DTYPE)
                    b_orig_dev = original_bias.to(device, dtype=COMPUTE_DTYPE)

                    # Calculate weight error
                    weight_error = W_orig_dev - W_dequant_dev
                    
                    # Propagate error through the linear layer's matrix multiplication
                    # Output error: (N, C_out) = (N, C_in) @ (C_in, C_out).T
                    output_error = X_calib_dev @ weight_error.T
                    
                    # The bias correction is the mean of this output error across the batch dimension
                    bias_correction = output_error.mean(dim=0)
                    
                    # Apply the correction to the original bias
                    b_new = b_orig_dev - bias_correction
                    
                    # Store the new bias, converting back to original dtype and CPU
                    new_tensors[bias_key] = b_new.cpu().to(original_bias.dtype)
                    
                    if verbose:
                        print(f"  - Original bias mean: {original_bias.mean().item():.6f}")
                        print(f"  - New bias mean     : {new_tensors[bias_key].mean().item():.6f}")
                    
                    # Clean up GPU memory
                    del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new

            # if t5xxl:
                # scale_input_key = f"{base_name}.scale_input" #this i the same as scale_weight - so what's the point?
                # new_tensors[scale_input_key] = dequant_scale.detach().clone().to(SCALE_DTYPE)

            if verbose:
                # if dequant_scale is not None:
                    # print(f"  - Dequant Scale  : {dequant_scale.item():.9}")
                print(f"  - Weight  : {quantized_fp8_tensor}")

        # Combine original non-weight tensors with new/modified ones
        for key, tensor in tensors.items():
            if key not in new_tensors:
                new_tensors[key] = tensor
                if verbose:
                    print(f"(+) Adding original non-quantized tensor: {key}")


    print("-" * 40)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}")
        return ""

    print("-" * 40)
    print("Summary:")
    print(f"  - Original tensor count : {len(tensors)}")
    print(f"  - Weights processed     : {processed_count}")
    print(f"  - Weights skipped       : {skipped_count}")
    print(f"  - Final tensor count    : {len(new_tensors)}")
    print("-" * 40)

    return f"Saved {len(new_tensors)} tensors to {output_file}"


def fp8_scaled_conversion(input:str, avoid_keys:str, remove_keys:str, mode:str="fp8_scaled", verbose=False, t5_compatibility:bool=True, row=True):
    calib_samples = 3072    # Number of random samples for calibration (bias correction)
    iterations = 50        # Number of optimization iterations per tensor row
    top_k = 1               # Number of optimization iterations per tensor.
    keep_distillation = True    # Exclude distillation layers from quantization.

    if input[0] in ["\"", "\'"] and input[0] == input[-1]:
        input = input[1:-1]

    if not os.path.exists(input):
        print(f"Error: Input file not found: {input}")
        return

    # Check for FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("Error: This version of PyTorch or this hardware does not support torch.float8_e4m3fn.")
        return

    fp8_type_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
    distill_str = "_nodistill" if keep_distillation else ""
    base_name = os.path.splitext(input)[0]
    match mode:
        case "fp16":
            output_file = f"{base_name}_fp16.safetensors"
        case "fp8_e4m3fn":
            output_file = f"{base_name}_{fp8_type_str}.safetensors"
        case "fp8_scaled":
            output_file = f"{base_name}_fp8s{'r' if row else ''}.safetensors"
        case "int8":
            output_file = f"{base_name}_int8r.safetensors"
        case "mixed8":
            output_file = f"{base_name}_m8r.safetensors"

    return convert_to_fp8_scaled(
        input,
        avoid_keys,
        remove_keys,
        mode,
        output_file,
        t5_compatibility,
        keep_distillation,
        calib_samples,
        iterations,
        top_k,
        verbose,
        row
    )


def convert_state_dict_to_fp8_scaled(tensors, keep_distillation: bool, calib_samples: int, iterations: int, top_k: int):
    # Check for FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        dtype_str = str(TARGET_FP8_DTYPE)
        print(f"Error: This version of PyTorch or this hardware does not support {dtype_str}.")
        return

    key_prefix = os.path.commonprefix(list(tensors.keys()))

    # Instantiate the converter with hyperparameters from command line
    converter = LearnedRoundingConverter(iterations, top_k)

    # generate calibration data for each unique input dimension as each dimension is found
    calibration_data_cache = {}

    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight')])
    total_weights = len(weight_keys)

    for i, key in enumerate(weight_keys):
        if keep_distillation and any(avoid_name in key for avoid_name in DISTILL_LAYER_KEYNAMES):
            continue

        print(f"({i+1}/{total_weights})", end="\r")

        original_tensor = tensors[key].clone()

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            tensors[key] = tensors[key].to(TARGET_FP8_DTYPE) # Store as empty FP8
            continue

        # Use the learned rounding converter
        quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert(original_tensor, False)

        # Store the results
        tensors[key] = quantized_fp8_tensor
        base_name = key[:-len('.weight')]
        bias_key = f"{base_name}.bias"
        if dequant_scale is not None:
            scale_weight_key = f"{base_name}.scale_weight"
            tensors[scale_weight_key] = dequant_scale.to(SCALE_DTYPE)

        # --- BIAS CORRECTION ---
        if bias_key in tensors:
            with torch.no_grad():
                original_bias = tensors[bias_key].clone()

                in_features = original_tensor.shape[1]
                if in_features not in calibration_data_cache:
                    calibration_data_cache[in_features] = torch.randn(calib_samples, in_features, dtype=COMPUTE_DTYPE)

                calibration_data = calibration_data_cache[in_features]
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Move tensors to the compute device
                W_orig_dev = original_tensor.to(device, dtype=COMPUTE_DTYPE)
                W_dequant_dev = dequantized_weight_tensor.to(device, dtype=COMPUTE_DTYPE)
                X_calib_dev = calibration_data.to(device, dtype=COMPUTE_DTYPE)
                b_orig_dev = original_bias.to(device, dtype=COMPUTE_DTYPE)

                # Calculate weight error
                weight_error = W_orig_dev - W_dequant_dev
                
                # Propagate error through the linear layer's matrix multiplication
                # Output error: (N, C_out) = (N, C_in) @ (C_in, C_out).T
                output_error = X_calib_dev @ weight_error.T
                
                # The bias correction is the mean of this output error across the batch dimension
                bias_correction = output_error.mean(dim=0)
                
                # Apply the correction to the original bias
                b_new = b_orig_dev - bias_correction
                
                # Store the new bias, converting back to original dtype and CPU
                tensors[bias_key] = b_new.cpu().to(original_bias.dtype)

                # Clean up GPU memory
                del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new

    print(f"({total_weights}/{total_weights})")

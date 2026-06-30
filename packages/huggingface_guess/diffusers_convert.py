import re
import torch

# conversion code from https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py

# =================#
# UNet Conversion  #    UNUSED
# =================#

# unet_conversion_map = (
# #   (  stable-diffusion,          HF Diffusers)
    # ("time_embed.0.weight",     "time_embedding.linear_1.weight"),
    # ("time_embed.0.bias",       "time_embedding.linear_1.bias"),
    # ("time_embed.2.weight",     "time_embedding.linear_2.weight"),
    # ("time_embed.2.bias",       "time_embedding.linear_2.bias"),
    # ("input_blocks.0.0.weight", "conv_in.weight"),
    # ("input_blocks.0.0.bias",   "conv_in.bias"),
    # ("out.0.weight",            "conv_norm_out.weight"),
    # ("out.0.bias",              "conv_norm_out.bias"),
    # ("out.2.weight",            "conv_out.weight"),
    # ("out.2.bias",              "conv_out.bias"),
# )

# unet_conversion_map_resnet = (
# #   (  stable-diffusion,  HF Diffusers)
    # ("in_layers.0",     "norm1"),
    # ("in_layers.2",     "conv1"),
    # ("out_layers.0",    "norm2"),
    # ("out_layers.3",    "conv2"),
    # ("emb_layers.1",    "time_emb_proj"),
    # ("skip_connection", "conv_shortcut"),
# )

# # expanded for clarity
# unet_conversion_map_layer = (
# #   (  stable-diffusion,      HF Diffusers)
    # ("input_blocks.1.0.",   "down_blocks.0.resnets.0."),
    # ("input_blocks.2.0.",   "down_blocks.0.resnets.1."),
    # ("input_blocks.4.0.",   "down_blocks.1.resnets.0."),
    # ("input_blocks.5.0.",   "down_blocks.1.resnets.1."),
    # ("input_blocks.7.0.",   "down_blocks.2.resnets.0."),
    # ("input_blocks.8.0.",   "down_blocks.2.resnets.1."),
    # ("input_blocks.10.0.",  "down_blocks.3.resnets.0."),
    # ("input_blocks.11.0.",  "down_blocks.3.resnets.1."),

    # ("input_blocks.1.1.",   "down_blocks.0.attentions.0."),
    # ("input_blocks.2.1.",   "down_blocks.0.attentions.1."),
    # ("input_blocks.4.1.",   "down_blocks.1.attentions.0."),
    # ("input_blocks.5.1.",   "down_blocks.1.attentions.1."),
    # ("input_blocks.7.1.",   "down_blocks.2.attentions.0."),
    # ("input_blocks.8.1.",   "down_blocks.2.attentions.1."),

    # ("output_blocks.0.0.",  "up_blocks.0.resnets.0."),
    # ("output_blocks.1.0.",  "up_blocks.0.resnets.1."),
    # ("output_blocks.2.0.",  "up_blocks.0.resnets.2."),
    # ("output_blocks.3.0.",  "up_blocks.1.resnets.0."),
    # ("output_blocks.4.0.",  "up_blocks.1.resnets.1."),
    # ("output_blocks.5.0.",  "up_blocks.1.resnets.2."),
    # ("output_blocks.6.0.",  "up_blocks.2.resnets.0."),
    # ("output_blocks.7.0.",  "up_blocks.2.resnets.1."),
    # ("output_blocks.8.0.",  "up_blocks.2.resnets.2."),
    # ("output_blocks.9.0.",  "up_blocks.3.resnets.0."),
    # ("output_blocks.10.0.", "up_blocks.3.resnets.1."),
    # ("output_blocks.11.0.", "up_blocks.3.resnets.2."),

    # ("output_blocks.3.1.",  "up_blocks.1.attentions.0."),
    # ("output_blocks.4.1.",  "up_blocks.1.attentions.1."),
    # ("output_blocks.5.1.",  "up_blocks.1.attentions.2."),
    # ("output_blocks.6.1.",  "up_blocks.2.attentions.0."),
    # ("output_blocks.7.1.",  "up_blocks.2.attentions.1."),
    # ("output_blocks.8.1.",  "up_blocks.2.attentions.2."),
    # ("output_blocks.9.1.",  "up_blocks.3.attentions.0."),
    # ("output_blocks.10.1.", "up_blocks.3.attentions.1."),
    # ("output_blocks.11.1.", "up_blocks.3.attentions.2."),

    # ("input_blocks.3.0.op.", "down_blocks.0.downsamplers.0.conv."),
    # ("input_blocks.6.0.op.", "down_blocks.1.downsamplers.0.conv."),
    # ("input_blocks.9.0.op.", "down_blocks.2.downsamplers.0.conv."),

    # ("output_blocks.2.1.", "up_blocks.0.upsamplers.0."),
    # ("output_blocks.5.2.", "up_blocks.1.upsamplers.0."),
    # ("output_blocks.8.2.", "up_blocks.2.upsamplers.0."),

    # ("middle_block.1.", "mid_block.attentions.0."),

    # ("middle_block.0.", "mid_block.resnets.0."),
    # ("middle_block.2.", "mid_block.resnets.1."),
# )

# def convert_unet_state_dict(unet_state_dict):
    # mapping = {k: k for k in unet_state_dict.keys()}

    # for sd_name, hf_name in unet_conversion_map:
        # mapping[hf_name] = sd_name

    # for k, v in mapping.items():
        # if "resnets" in k:
            # for sd_part, hf_part in unet_conversion_map_resnet:
                # v = v.replace(hf_part, sd_part)
            # mapping[k] = v

    # for k, v in mapping.items():
        # for sd_part, hf_part in unet_conversion_map_layer:
            # v = v.replace(hf_part, sd_part)
        # mapping[k] = v

    # new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

    # return new_state_dict


# ================#
# VAE Conversion  #    USED in backend.loader
# ================#

# expanded for clarity
vae_conversion_map = (
#     stable-diffusion           HF Diffusers)
    ('nin_shortcut',            'conv_shortcut'),
    ('norm_out',                'conv_norm_out'),

    ('mid.attn_1.',             'mid_block.attentions.0.'),
    ('mid.block_1.',            'mid_block.resnets.0.'),
    ('mid.block_2.',            'mid_block.resnets.1.'),

    ('encoder.down.0.block.0.', 'encoder.down_blocks.0.resnets.0.'),
    ('encoder.down.0.block.1.', 'encoder.down_blocks.0.resnets.1.'),
    ('encoder.down.1.block.0.', 'encoder.down_blocks.1.resnets.0.'),
    ('encoder.down.1.block.1.', 'encoder.down_blocks.1.resnets.1.'),
    ('encoder.down.2.block.0.', 'encoder.down_blocks.2.resnets.0.'),
    ('encoder.down.2.block.1.', 'encoder.down_blocks.2.resnets.1.'),
    ('encoder.down.3.block.0.', 'encoder.down_blocks.3.resnets.0.'),
    ('encoder.down.3.block.1.', 'encoder.down_blocks.3.resnets.1.'),

    ('down.0.downsample.',      'down_blocks.0.downsamplers.0.'),
    ('down.1.downsample.',      'down_blocks.1.downsamplers.0.'),
    ('down.2.downsample.',      'down_blocks.2.downsamplers.0.'),
    ('up.3.upsample.',          'up_blocks.0.upsamplers.0.'),
    ('up.2.upsample.',          'up_blocks.1.upsamplers.0.'),
    ('up.1.upsample.',          'up_blocks.2.upsamplers.0.'),

    ('decoder.up.3.block.0.',   'decoder.up_blocks.0.resnets.0.'),
    ('decoder.up.3.block.1.',   'decoder.up_blocks.0.resnets.1.'),
    ('decoder.up.3.block.2.',   'decoder.up_blocks.0.resnets.2.'),
    ('decoder.up.2.block.0.',   'decoder.up_blocks.1.resnets.0.'),
    ('decoder.up.2.block.1.',   'decoder.up_blocks.1.resnets.1.'),
    ('decoder.up.2.block.2.',   'decoder.up_blocks.1.resnets.2.'),
    ('decoder.up.1.block.0.',   'decoder.up_blocks.2.resnets.0.'),
    ('decoder.up.1.block.1.',   'decoder.up_blocks.2.resnets.1.'),
    ('decoder.up.1.block.2.',   'decoder.up_blocks.2.resnets.2.'),
    ('decoder.up.0.block.0.',   'decoder.up_blocks.3.resnets.0.'),
    ('decoder.up.0.block.1.',   'decoder.up_blocks.3.resnets.1.'),
    ('decoder.up.0.block.2.',   'decoder.up_blocks.3.resnets.2.'),
)

vae_conversion_map_attn = (
#     SD           HF
    ("norm.",     "group_norm."),
    ("q.",        "query."),
    ("k.",        "key."),
    ("v.",        "value."),
    ("q.",        "to_q."),
    ("k.",        "to_k."),
    ("v.",        "to_v."),
    ("proj_out.", "to_out.0."),
    ("proj_out.", "proj_attn."),
)


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}

    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v

    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v

    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                if k.endswith(".conv.weight") and v.ndim == 5:
                    new_state_dict[k] = v.reshape(*v.shape, 1, 1, 1)
                else:
                    new_state_dict[k] = v.reshape(*v.shape, 1, 1)
    return new_state_dict


# =========================#
# Text Encoder Conversion  #    USED in huggingface_guess.model_list for saving of CLIP-G and CLIP-H
# =========================#

textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}

# This function exists because at the time of writing torch.cat can't do fp8 with cuda
def cat_tensors(tensors):
    x = 0
    for t in tensors:
        x += t.shape[0]

    shape = [x] + list(tensors[0].shape)[1:]
    out = torch.empty(shape, device=tensors[0].device, dtype=tensors[0].dtype)

    x = 0
    for t in tensors:
        out[x:x + t.shape[0]] = t
        x += t.shape[0]

    return out

def convert_text_enc_state_dict_v20(text_enc_dict, prefix=""):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if not k.startswith(prefix):
            continue
        if (
                k.endswith(".self_attn.q_proj.weight")
                or k.endswith(".self_attn.k_proj.weight")
                or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
                k.endswith(".self_attn.q_proj.bias")
                or k.endswith(".self_attn.k_proj.bias")
                or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        text_proj = "transformer.text_projection.weight"
        if k.endswith(text_proj):
            new_state_dict[k.replace(text_proj, "text_projection")] = v.transpose(0, 1).contiguous()
        else:
            relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
            new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = cat_tensors(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = cat_tensors(tensors)

    return new_state_dict


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict

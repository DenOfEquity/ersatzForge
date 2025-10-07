# https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/IPAdapterPlus.py from some early version
# Then maintained by Forge to add InstanceID and many other things


import torch

import os
import math

from backend import memory_management, attention, utils
from backend.misc.image_resize import contrast_adaptive_sharpening

from backend.patcher.clipvision import clip_preprocess
from modules_forge.shared import controlnet_dir, models_path

from torch import nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT

from lib_ipadapter.resampler import PerceiverAttention, FeedForward, Resampler

GLOBAL_MODELS_DIR = os.path.join(models_path, "ipadapter")
MODELS_DIR = GLOBAL_MODELS_DIR
INSIGHTFACE_DIR = os.path.join(models_path, "insightface")


class FacePerceiverResampler(torch.nn.Module):
    def __init__(
            self,
            *,
            dim=768,
            depth=4,
            dim_head=64,
            heads=16,
            embedding_dim=1280,
            output_dim=768,
            ff_mult=4,
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class MLPProjModelFaceId(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        clip_extra_context_tokens = self.proj(id_embeds)
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(-1, self.num_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class ProjModelFaceIdPlus(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, scale=1.0, shortcut=False):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class To_KV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value


def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)


def image_add_noise(image, noise):
    image = image.permute([0, 3, 1, 2])
    torch.manual_seed(0)  # use a fixed random for reproducible results
    transforms = TT.Compose([
        TT.CenterCrop(min(image.shape[2], image.shape[3])),
        TT.Resize((224, 224), interpolation=TT.InterpolationMode.BICUBIC, antialias=True),
        TT.ElasticTransform(alpha=75.0, sigma=noise * 3.5),  # shuffle the image
        TT.RandomVerticalFlip(p=1.0),  # flip the image to change the geometry even more
        TT.RandomHorizontalFlip(p=1.0),
    ])
    image = transforms(image.cpu())
    image = image.permute([0, 2, 3, 1])
    image = image + ((0.25 * (1 - noise) + 0.05) * torch.randn_like(image))  # add further random noise
    return image


def zeroed_hidden_states(clip_vision, batch_size):
    image = torch.zeros([batch_size, 224, 224, 3])
    memory_management.load_model_gpu(clip_vision.patcher)
    pixel_values = clip_preprocess(image.to(clip_vision.load_device)).to(torch.float32)
    outputs = clip_vision.model(pixel_values=pixel_values, output_hidden_states=True)
    outputs = outputs.hidden_states[-2].to(memory_management.intermediate_device())
    return outputs


def tensorToNP(image):
    out = torch.clamp(255. * image.detach().cpu(), 0, 255).to(torch.uint8)
    out = out[..., [2, 1, 0]]
    out = out.numpy()

    return out


def NPToTensor(image):
    out = torch.from_numpy(image)
    out = torch.clamp(out.to(torch.float) / 255., 0.0, 1.0)
    out = out[..., [2, 1, 0]]

    return out


class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024,
                 clip_embeddings_dim=1024, clip_extra_context_tokens=4,
                 is_sdxl=False, is_plus=False, is_full=False,
                 is_faceid=False, is_instant_id=False):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus
        self.is_instant_id = is_instant_id

        if is_instant_id:
            self.image_proj_model = self.init_proj_instantid()
        elif is_faceid:
            self.image_proj_model = self.init_proj_faceid()
        elif is_plus:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(ipadapter_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
        if self.is_full:
            image_proj_model = MLPProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=self.clip_embeddings_dim
            )
        else:
            image_proj_model = Resampler(
                dim=self.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if self.is_sdxl else 12,
                num_queries=self.clip_extra_context_tokens,
                embedding_dim=self.clip_embeddings_dim,
                output_dim=self.output_cross_attention_dim,
                ff_mult=4
            )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=1280,
                num_tokens=4,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    def init_proj_instantid(self, image_emb_dim=512, num_tokens=16):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        )
        return image_proj_model

    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def get_image_embeds_faceid_plus(self, face_embed, clip_embed, s_scale, shortcut):
        embeds = self.image_proj_model(face_embed, clip_embed, scale=s_scale, shortcut=shortcut)
        return embeds

    def get_image_embeds_instantid(self, prompt_image_emb, noise):
        c = self.image_proj_model(prompt_image_emb)
        torch.manual_seed(0)  # use a fixed random for reproducible results
        uc = self.image_proj_model(torch.randn_like(prompt_image_emb) * prompt_image_emb.std() * noise)
        return c, uc


class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, ipadapter, number, cond, uncond, weight_type, mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.number = number
        self.weight_type = [weight_type]
        self.masks = [mask]
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]
        self.unfold_batch = [unfold_batch]

        self.k_key = str(self.number * 2 + 1) + "_to_k_ip"
        self.v_key = str(self.number * 2 + 1) + "_to_v_ip"

    def set_new_condition(self, weight, ipadapter, number, cond, uncond, weight_type, mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.weight_type.append(weight_type)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)
        self.unfold_batch.append(unfold_batch)

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]

        sigma = extra_options["sigmas"][0] if 'sigmas' in extra_options else None
        sigma = sigma.item() if sigma is not None else 999999999.9

        # extra options for AnimateDiff
        ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

        q = n
        k = context_attn2
        v = value_attn2
        b = q.shape[0]
        qs = q.shape[1]
        batch_prompt = b // len(cond_or_uncond)
        out = attention.attention_function(q, k, v, extra_options["n_heads"])
        _, _, lh, lw = extra_options["original_shape"]

        for weight, cond, uncond, ipadapter, mask, weight_type, sigma_start, sigma_end, unfold_batch in zip(self.weights, self.conds, self.unconds, self.ipadapters, self.masks, self.weight_type, self.sigma_start, self.sigma_end, self.unfold_batch):
            if sigma > sigma_start or sigma < sigma_end:
                continue

            if unfold_batch and cond.shape[0] > 1:
                if 1 in cond_or_uncond:     #uncond
                    # Check AnimateDiff context window
                    if ad_params is not None and ad_params["sub_idxs"] is not None:
                        if cond.shape[0] >= ad_params["full_length"]:    # if images length matches or exceeds full_length get sub_idx images
                            uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
                        else:    # otherwise, expand by repeating last
                            uncond = torch.cat((uncond, uncond[-1:].repeat((ad_params["full_length"] - uncond.shape[0], 1, 1))), dim=0)
                            uncond = uncond[ad_params["sub_idxs"]]

                    if cond.shape[0] < batch_prompt:    # if we don't have enough reference images repeat the last one
                        uncond = torch.cat((uncond, uncond[-1:].repeat((batch_prompt - uncond.shape[0], 1, 1))), dim=0)
                    elif cond.shape[0] > batch_prompt:    # if we have too many remove the excess
                        uncond = uncond[:batch_prompt]

                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond)

                if 0 in cond_or_uncond:     #cond
                    if ad_params is not None and ad_params["sub_idxs"] is not None:
                        if cond.shape[0] >= ad_params["full_length"]:
                            cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                        else:
                            cond = torch.cat((cond, cond[-1:].repeat((ad_params["full_length"] - cond.shape[0], 1, 1))), dim=0)
                            cond = cond[ad_params["sub_idxs"]]

                    if cond.shape[0] < batch_prompt:
                        cond = torch.cat((cond, cond[-1:].repeat((batch_prompt - cond.shape[0], 1, 1))), dim=0)
                    elif cond.shape[0] > batch_prompt:
                        cond = cond[:batch_prompt]

                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond)

            else:
                if 1 in cond_or_uncond:
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond).repeat(batch_prompt, 1, 1)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond).repeat(batch_prompt, 1, 1)
                if 0 in cond_or_uncond:
                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond).repeat(batch_prompt, 1, 1)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond).repeat(batch_prompt, 1, 1)

            if cond_or_uncond == [1]:
                ip_k = k_uncond
                ip_v = v_uncond
            elif cond_or_uncond == [0]:
                ip_k = k_cond
                ip_v = v_cond
            else:   # should only ever be [1, 0]
                ip_k = torch.cat([k_uncond, k_cond], dim=0)
                ip_v = torch.cat([v_uncond, v_cond], dim=0)

            if weight_type.startswith("linear"):
                ip_k *= weight
                ip_v *= weight
            else:
                if weight_type.startswith("channel"):
                    # code by Lvmin Zhang at Stanford University as also seen on Fooocus IPAdapter implementation
                    # please read licensing notes https://github.com/lllyasviel/Fooocus/blob/69a23c4d60c9e627409d0cb0f8862cdb015488eb/extras/ip_adapter.py#L234
                    ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                    ip_v_offset = ip_v - ip_v_mean
                    _, _, C = ip_k.shape
                    channel_penalty = float(C) / 1280.0
                    W = weight * channel_penalty
                    ip_k = ip_k * W
                    ip_v = ip_v_offset + ip_v_mean * W

            out_ip = attention.attention_function(q, ip_k.to(org_dtype), ip_v.to(org_dtype), extra_options["n_heads"])
            
            if weight_type.startswith("original"):
                out_ip = out_ip * weight

            if mask is not None:
                # TODO: needs checking
                mask_h = lh / math.sqrt(lh * lw / qs)
                mask_h = int(mask_h) + int((qs % int(mask_h)) != 0)
                mask_w = qs // mask_h

                # check if using AnimateDiff and sliding context window
                if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
                    # if mask length matches or exceeds full_length, just get sub_idx masks, resize, and continue
                    if mask.shape[0] >= ad_params["full_length"]:
                        mask_downsample = torch.Tensor(mask[ad_params["sub_idxs"]])
                        mask_downsample = F.interpolate(mask_downsample.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
                    # otherwise, need to do more to get proper sub_idxs masks
                    else:
                        # resize to needed attention size (to save on memory)
                        mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
                        # check if mask length matches full_length - if not, make it match
                        if mask_downsample.shape[0] < ad_params["full_length"]:
                            mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:].repeat((ad_params["full_length"] - mask_downsample.shape[0], 1, 1))), dim=0)
                        # if we have too many remove the excess (should not happen, but just in case)
                        if mask_downsample.shape[0] > ad_params["full_length"]:
                            mask_downsample = mask_downsample[:ad_params["full_length"]]
                        # now, select sub_idxs masks
                        mask_downsample = mask_downsample[ad_params["sub_idxs"]]
                # otherwise, perform usual mask interpolation
                else:
                    mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)

                # if we don't have enough masks repeat the last one until we reach the right size
                if mask_downsample.shape[0] < batch_prompt:
                    mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:, :, :].repeat((batch_prompt - mask_downsample.shape[0], 1, 1))), dim=0)
                # if we have too many remove the exceeding
                elif mask_downsample.shape[0] > batch_prompt:
                    mask_downsample = mask_downsample[:batch_prompt, :, :]

                # repeat the masks
                mask_downsample = mask_downsample.repeat(len(cond_or_uncond), 1, 1)
                mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                out_ip.mul_(mask_downsample)

            out.add_(out_ip)

        return out.to(dtype=org_dtype)


insightface_face_align = None


class InsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"],),
            },
        }

    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load_insight_face"
    CATEGORY = "ipadapter"

    def load_insight_face(self, name="buffalo_l", provider="CPU"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise Exception(e)

        if torch.cuda.is_available():
            provider = "CUDA"

        if name == 'antelopev2':
            from modules.modelloader import load_file_from_url
            model_root = os.path.join(INSIGHTFACE_DIR, 'models', "antelopev2")
            if not model_root:
                os.makedirs(model_root, exist_ok=True)
            for local_file, url in (
                    ("1k3d68.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/1k3d68.onnx"),
                    ("2d106det.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/2d106det.onnx"),
                    ("genderage.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/genderage.onnx"),
                    ("glintr100.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx"),
                    ("scrfd_10g_bnkps.onnx",
                     "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"),
            ):
                local_path = os.path.join(model_root, local_file)
                if not os.path.exists(local_path):
                    load_file_from_url(url, model_dir=model_root)

        from insightface.utils import face_align
        global insightface_face_align
        insightface_face_align = face_align

        model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider', ])
        model.prepare(ctx_id=0, det_size=(640, 640))

        return model


class IPAdapterApply:
    def apply_ipadapter(self, ipadapter, model, weight, clip_vision=None, images=None, weight_type="original",
                        noise=0.0, sharpening=0.0, embeds=None, attn_mask=None, start_at=0.0, end_at=1.0, unfold_batch=False,
                        insightface=None, faceid_v2=False, weight_v2=False, instant_id=False):

        self.dtype = torch.float16 if memory_management.should_use_fp16(prioritize_performance=False, manual_cast=True) else torch.float32
        self.device = memory_management.get_torch_device()
        self.weight = weight
        self.is_full = "proj.3.weight" in ipadapter["image_proj"]
        self.is_portrait = "proj.2.weight" in ipadapter["image_proj"] and "proj.3.weight" not in ipadapter["image_proj"] and "0.to_q_lora.down.weight" not in ipadapter["ip_adapter"]
        self.is_faceid = self.is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
        self.is_plus = (self.is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"])
        self.is_instant_id = instant_id

        if (self.is_faceid or self.is_instant_id) and not insightface:
            raise Exception('InsightFace must be provided for FaceID/InstantID models.')

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280 if self.is_plus and self.is_sdxl and not self.is_faceid else output_cross_attention_dim
        clip_extra_context_tokens = 16 if self.is_plus or self.is_portrait else 4

        if self.is_instant_id:
            cross_attention_dim = output_cross_attention_dim

        if embeds is not None:
            embeds = torch.unbind(embeds)
            clip_embed = embeds[0].cpu()
            clip_embed_zeroed = embeds[1].cpu()
        else:
            if self.is_instant_id:
                face_embed = []

                for i in range(len(images)):
                    if isinstance(images[i], list):
                        images[i] = images[i][0]

                    face_img = self.prep_image(images[i], sharpening)

                    for size in [(size, size) for size in range(640, 128, -64)]:
                        insightface.det_model.input_size = size
                        face = insightface.get(face_img[0])
                        if face:
                            if len(face) > 1:   # only use the largest face
                                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
 
                            if isinstance(face, list):
                                face = face[0]

                            embed = face.get('embedding', None)
                            if embed is not None:
                                face_embed.append(torch.from_numpy(embed).unsqueeze(0))
                            break
                    else:
                        raise Exception('InsightFace: No face detected.')

                face_embed = torch.stack(face_embed, dim=0).mean(dim=0, keepdim=True)
                clip_embed = face_embed

            elif self.is_faceid:
                face_embed = []
                face_clipvision = []

                for i in range(len(images)):
                    if isinstance(images[i], list):
                        images[i] = images[i][0]

                    face_img = self.prep_image(images[i], sharpening)

                    for size in [(size, size) for size in range(640, 128, -64)]:
                        insightface.det_model.input_size = size  # TODO: hacky but seems to be working
                        face = insightface.get(face_img[0])
                        if face:
                            if len(face) > 1:   # only use the largest face
                                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]

                            if isinstance(face, list):
                                face = face[0]

                            face_embed.append(torch.from_numpy(face.normed_embedding).unsqueeze(0))
                            face_clipvision.append(NPToTensor(insightface_face_align.norm_crop(face_img[0], landmark=face.kps, image_size=224)))
                            break
                    else:
                        raise Exception('InsightFace: No face detected.')

                face_embed = torch.stack(face_embed, dim=0)
                image = torch.stack(face_clipvision, dim=0)

                neg_image = image_add_noise(image, noise) if noise > 0 else None

                if self.is_plus:
                    clip_embed = clip_vision.encode_image(image).penultimate_hidden_states
                    if noise > 0:
                        clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
                    else:
                        clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])

                    face_embed_zeroed = torch.zeros_like(face_embed)
                else:
                    clip_embed = face_embed
                    clip_embed_zeroed = torch.zeros_like(clip_embed)
            else:
                clip_embeds = []
                zero_embeds = []
                for i in range(len(images)):
                    if isinstance(images[i], list):
                        images[i] = images[i][0]

                    image = self.prep_image(images[i], sharpening, convertNP=False)

                    clip_embed = clip_vision.encode_image(image)
                    neg_image = image_add_noise(image, noise) if noise > 0 else None

                    if self.is_plus:
                        clip_embed = clip_embed.penultimate_hidden_states
                        if noise > 0:
                            clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
                        else:
                            clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
                    else:
                        clip_embed = clip_embed.image_embeds
                        if noise > 0:
                            clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
                        else:
                            clip_embed_zeroed = torch.zeros_like(clip_embed)

                    clip_embeds.append(clip_embed)
                    zero_embeds.append(clip_embed_zeroed)
                    
                    clip_embed = torch.cat(clip_embeds, dim=0)
                    clip_embed_zeroed = torch.cat(zero_embeds, dim=0)

        clip_embeddings_dim = clip_embed.shape[-1]

        self.ipadapter = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=self.is_sdxl,
            is_plus=self.is_plus,
            is_full=self.is_full,
            is_faceid=self.is_faceid,
            is_instant_id=self.is_instant_id
        )

        self.ipadapter.to(self.device, dtype=self.dtype)

        if self.is_instant_id:
            image_prompt_embeds, uncond_image_prompt_embeds = self.ipadapter.get_image_embeds_instantid(face_embed.to(self.device, dtype=self.dtype), noise)
        elif self.is_faceid and self.is_plus:
            image_prompt_embeds = self.ipadapter.get_image_embeds_faceid_plus(face_embed.to(self.device, dtype=self.dtype), clip_embed.to(self.device, dtype=self.dtype), weight_v2, faceid_v2)
            uncond_image_prompt_embeds = self.ipadapter.get_image_embeds_faceid_plus(face_embed_zeroed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype), weight_v2, faceid_v2)
        else:
            image_prompt_embeds, uncond_image_prompt_embeds = self.ipadapter.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        if self.is_instant_id:
            def modifier(cnet, x_noisy, t, cond, batched_number):
                cond_mark = cond['transformer_options']['cond_mark'][:, None, None].to(cond['c_crossattn'])  # cond is 0
                c_crossattn = image_prompt_embeds * (1.0 - cond_mark) + uncond_image_prompt_embeds * cond_mark
                cond['c_crossattn'] = c_crossattn
                return x_noisy, t, cond, batched_number

            model.add_controlnet_conditioning_modifier(modifier)

        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        sigma_start = model.model.predictor.percent_to_sigma(start_at)
        sigma_end = model.model.predictor.percent_to_sigma(end_at)

        patch_kwargs = {
            "number": 0,
            "weight": self.weight,
            "ipadapter": self.ipadapter,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": unfold_batch,
        }

#patch different blocks for style/composition - see Cubiq
        if not self.is_sdxl:
            for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
                set_model_patch_replace(model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # id of output_blocks that have cross attention
                set_model_patch_replace(model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(model, patch_kwargs, ("middle", 0))
        else:
            for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
                patch_kwargs["number"] += 1

        return model

    def prep_image(self, image, sharpening, convertNP=True, channels_last=True):
        if sharpening > 0.0:
            image = image ** 2.2
            image = contrast_adaptive_sharpening(image, sharpening, channels_last)
            image = image ** (1/2.2)
        
        if convertNP:
            return tensorToNP(image)
        else:
            return image


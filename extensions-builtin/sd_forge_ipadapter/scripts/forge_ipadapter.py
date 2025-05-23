from modules_forge.supported_preprocessor import Preprocessor, PreprocessorClipVision, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import numpy_to_pytorch
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_ipadapter.IPAdapterPlus import IPAdapterApply, InsightFaceLoader
from pathlib import Path
import random

cached_insightfaceA = None  # antelopev2
cached_insightface = None   # buffalo_l


class PreprocessorForInstantID(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.slider_resolution = PreprocessorParameter(
            label='Tiles (n * n) (limited by source(s))', minimum=1, maximum=16, value=1, step=1, visible=True)
        self.slider_1 = PreprocessorParameter(label='Noise', minimum=0.0, maximum=1.0, value=0.23, step=0.01, visible=True)
        self.slider_2 = PreprocessorParameter(label='Sharpening', minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True)
        self.tags = ['Instant-ID']
        self.model_filename_filters = ['Instant-ID', 'Instant_ID']
        self.sorting_priority = 20
        self.model = None

    def load_insightface(self):
        global cached_insightfaceA
        if cached_insightfaceA is None:
            cached_insightfaceA = InsightFaceLoader().load_insight_face(name="antelopev2")[0]
        return cached_insightfaceA


    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=0.0, **kwargs):
        cond = dict(
            clip_vision=None,
            insightface=self.load_insightface(),
            image=input_image,
            weight_type="original",
            noise=slider_1,
            sharpening=slider_2,
            embeds=None,
            unfold_batch=False,
            tiles=resolution,
            instant_id=True,
        )
        return cond


add_supported_preprocessor(PreprocessorForInstantID(
    name='Insightface (Instant-ID)',
))


class PreprocessorForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.slider_resolution = PreprocessorParameter(
            label='Tiles (n * n) (limited by source(s))', minimum=1, maximum=16, value=1, step=1, visible=True)
        self.slider_1 = PreprocessorParameter(label='Noise', minimum=0.0, maximum=1.0, value=0.23, step=0.01, visible=True)
        self.slider_2 = PreprocessorParameter(label='Sharpening', minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True)
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']
        self.sorting_priority = 20

    def load_insightface(self):
        global cached_insightface
        if cached_insightface is None:
            cached_insightface = InsightFaceLoader().load_insight_face()[0]
        return cached_insightface

    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=0.0, **kwargs):
        cond = dict(
            clip_vision=None if '(Portrait)' in self.name else self.load_clipvision(),
            insightface=self.load_insightface() if 'InsightFace' in self.name else None,
            image=input_image,
            weight_type="original",
            noise=slider_1,
            sharpening=slider_2,
            embeds=None,
            unfold_batch=False,
            tiles=resolution,
        )
        return cond


add_supported_preprocessor(PreprocessorForIPAdapter(
    name='CLIP-H-Face (Ostris) (IPAdapter)',
    url='https://huggingface.co/ostris/CLIP-H-Face-v3/resolve/main/model.safetensors',
    filename='CLIP-H-Face-v3.safetensors'
))
add_supported_preprocessor(PreprocessorForIPAdapter(
    name='InsightFace+CLIP-H-Face (Ostris) (IPAdapter)',
    url='https://huggingface.co/ostris/CLIP-H-Face-v3/resolve/main/model.safetensors',
    filename='CLIP-H-Face-v3.safetensors'
))


add_supported_preprocessor(PreprocessorForIPAdapter(
    name='CLIP-ViT-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))
add_supported_preprocessor(PreprocessorForIPAdapter(
    name='InsightFace+CLIP-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))


add_supported_preprocessor(PreprocessorForIPAdapter(
    name='CLIP-ViT-bigG (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))
add_supported_preprocessor(PreprocessorForIPAdapter(
    name='InsightFace+CLIP-G (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))


add_supported_preprocessor(PreprocessorForIPAdapter(
    name='InsightFace (IPAdapter) (Portrait)',
    url='',
    filename=''
))


#make new precossor for instantid, combined keypoints and eembedding
# _call with input image, get keypoints and embedding, pass to ipadapter functiom
# no extra processing necessay
# where to put keypoints? cross-attn

class IPAdapterPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        model = state_dict

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        if "ip_adapter" not in model.keys() or len(model["ip_adapter"]) == 0:
            return None

        o = IPAdapterPatcher(model)

        model_filename = Path(ckpt_path).name.lower()
        if 'v2' in model_filename:
            o.faceid_v2 = True
            o.weight_v2 = True

        return o

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        self.faceid_v2 = False
        self.weight_v2 = False
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        if isinstance(cond, list):  # should always be True
            images = []
            for c in cond:
                tile_count = c['tiles']

                image = c['image']
                
                if c['insightface']:
                    tile_type = 'Interleaved'
                elif c['clip_vision']:
                    tile_type = 'Tiled'

                if tile_count > 1:
                    r = min(image.shape[0] // 128, image.shape[1] // 128)
                    r = min(r, tile_count)
                    if tile_type == 'Interleaved' and r >= 2:   #interleaved split into r*r images
                        for i in range(r):
                            for j in range(r):
                                part = image[i::r, j::r]
                                images.append(numpy_to_pytorch(part))
                    elif tile_type == "Tiled":
                        if tile_count > 1:
                            images.append(numpy_to_pytorch(image))

                        tiles = []
                        tile_sizeX = max(image.shape[0] // tile_count, 224)
                        tile_sizeY = max(image.shape[1] // tile_count, 224)
                        for i in range(0, image.shape[0], tile_sizeX):
                            for j in range(0, image.shape[1], tile_sizeY):
                                tile = image[i:i+tile_sizeX, j:j+tile_sizeY]
                                tiles.append(numpy_to_pytorch(tile))

                        # random.shuffle(tiles)
                        images.append(tiles)
                    else:
                        images.append(numpy_to_pytorch(image))
                elif c['insightface'] or c['clip_vision']:
                    images.append(numpy_to_pytorch(image))
                else:
                    images.append(image)    # unused, but could be for already calculated embeds

            random.shuffle(images)

            pcond = cond[0].copy()
            pcond['images'] = images
            del pcond['tiles']
            del pcond['image']

        # else:
            # global cached_insightfaceA
            # if cached_insightfaceA is None:
                # cached_insightfaceA = InsightFaceLoader().load_insight_face(name="antelopev2")[0]

            # print (cond.shape, cond)

            # pcond = { 'insightface': cached_insightfaceA, 'images': cond, 'instant_id': True, }


        unet = IPAdapterApply().apply_ipadapter(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=self.faceid_v2,
            weight_v2=self.weight_v2,
            attn_mask=mask.squeeze(1) if mask is not None else None,
            **pcond,
        )[0]

        process.sd_model.forge_objects.unet = unet
        return




add_supported_control_model(IPAdapterPatcher)

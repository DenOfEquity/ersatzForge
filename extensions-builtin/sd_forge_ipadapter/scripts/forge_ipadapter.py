from modules_forge.supported_preprocessor import Preprocessor, PreprocessorClipVision, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import numpy_to_pytorch, resize_image_with_pad
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_ipadapter.IPAdapterPlus import IPAdapterApply, InsightFaceLoader
from pathlib import Path
import random, numpy, math
from cv2 import circle, ellipse2Poly, fillConvexPoly

cached_insightfaceA = None  # antelopev2
cached_insightface = None   # buffalo_l
cached_eva_clip = None

from pulid.PuLID import ApplyPulid
from pulid.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


# todo: caching results - currently stuff that could be preprocess are done during processing
#       dictionary of last n input images?, split for insightface and clip - but two types of insightface and 4 CLIPs currently supported
# todo: consolidate some shared code between pulid and ipadapter


class PreprocessorForInstantID(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if 'keypoints' in name: #resolution useful? maybe just set to reasonable value
            self.slider_resolution = PreprocessorParameter(label='Resolution', minimum=256, maximum=2048, value=512, step=64, visible=True)
            self.slider_1 = PreprocessorParameter(visible=False)
            self.slider_2 = PreprocessorParameter(visible=False)
            self.cache = None
            self.cacheHash = None
        else:
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
            cached_insightfaceA = InsightFaceLoader().load_insight_face(name="antelopev2")
        return cached_insightfaceA

    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=0.0, **kwargs):
        if 'keypoints' in self.name:
            insightface = self.load_insightface()

            def draw_kps(img: numpy.ndarray, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
                stickwidth = 4
                limbSeq = numpy.array([[0, 2], [1, 2], [3, 2], [4, 2]])
                kps = numpy.array(kps)

                h, w, _ = img.shape
                out_img = numpy.zeros([h, w, 3])

                for i in range(len(limbSeq)):
                    index = limbSeq[i]
                    color = color_list[index[0]]

                    x = kps[index][:, 0]
                    y = kps[index][:, 1]
                    length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
                    polygon = ellipse2Poly((int(numpy.mean(x)), int(numpy.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    out_img = fillConvexPoly(out_img.copy(), polygon, color)
                out_img = (out_img * 0.6).astype(numpy.uint8)

                for idx_kp, kp in enumerate(kps):
                    color = color_list[idx_kp]
                    x, y = kp
                    out_img = circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

                return out_img.astype(numpy.uint8)

            img, remove_pad = resize_image_with_pad(input_image, resolution)

            for size in [(size, size) for size in range(640, 128, -64)]:
                insightface.det_model.input_size = size
                face = insightface.get(img)
                if face:
                    if len(face) > 1: # only use the maximum face
                        face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]

                    if isinstance(face, list):
                        face = face[0]

                    result = remove_pad(draw_kps(img, face['kps']))

                    cond = result
                    break
                else:
                    raise Exception('InsightFace: No face detected.')

        else:
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
    name='Insightface (Instant-ID) embeds',
))
add_supported_preprocessor(PreprocessorForInstantID(
    name='Insightface (Instant-ID) keypoints',
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
            cached_insightface = InsightFaceLoader().load_insight_face()
        return cached_insightface

    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=0.0, **kwargs):
        cond = dict(
            clip_vision=None if '(Portrait)' in self.name else self.load_clipvision(),
            insightface=self.load_insightface() if 'InsightFace' in self.name else None,
            image=input_image,
            weight_type="channel",#original
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


class PreprocessorForPuLID(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.slider_resolution = PreprocessorParameter(
            label='Fidelity (lower = likeness closer to source)', minimum=0, maximum=32, value=1, step=1, visible=True)
        self.slider_1 = PreprocessorParameter(label='Noise', minimum=0.0, maximum=1.0, value=0.23, step=0.01, visible=True)
        self.slider_2 = PreprocessorParameter(label='Sharpening', minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True)
        self.tags = ['PuLID']
        self.model_filename_filters = ['PuLID']
        self.model = None
        self.sorting_priority = 20

        if self.name == 'PuLID (ortho)':
            self.method = 'ortho'
        elif self.name == 'PuLID (ortho_v2)':
            self.method = 'ortho_v2'
        else:
            self.method = 'default'


    def __call__(self, input_image, resolution, slider_1=0.23, slider_2=0.0, **kwargs):
        cond = dict(
            image=input_image,
            noise=slider_1,
            sharpening=slider_2,
            fidelity=resolution,
            method=self.method,
            pulid=True,
        )

        return cond

add_supported_preprocessor(PreprocessorForPuLID('PuLID'))
add_supported_preprocessor(PreprocessorForPuLID('PuLID (ortho)'))
add_supported_preprocessor(PreprocessorForPuLID('PuLID (ortho_v2)'))


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
            o.weight_v2 = 2.0

        return o

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        self.faceid_v2 = False
        self.weight_v2 = 0.0
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet.clone()

        if 'pulid' in cond[0] and cond[0]['pulid'] == True:
            if isinstance(cond, list):  # should always be True
                images = []
                for c in cond:
                    image = c['image']
                    images.append(numpy_to_pytorch(image))

            global cached_insightfaceA
            if cached_insightfaceA is None:
                cached_insightfaceA = InsightFaceLoader().load_insight_face(name="antelopev2")

            global cached_eva_clip
            if cached_eva_clip is None:
                from pulid.eva_clip.factory import create_model_and_transforms

                model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)

                model = model.visual

                eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
                eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
                if not isinstance(eva_transform_mean, (list, tuple)):
                    model["image_mean"] = (eva_transform_mean,) * 3
                if not isinstance(eva_transform_std, (list, tuple)):
                    model["image_std"] = (eva_transform_std,) * 3

                cached_eva_clip = model

            unet = ApplyPulid().apply_pulid(
                unet,
                self.ip_adapter,
                cached_eva_clip,
                cached_insightfaceA,
                images,
                self.strength,
                self.start_percent,
                self.end_percent,
                cond[0]['sharpening'],
                cond[0]['method'],  #'default', 'ortho' (fidelity), 'ortho_v2' (style)
                noise=cond[0]['noise'],
                fidelity=cond[0]['fidelity'],
                attn_mask=mask.squeeze(1) if mask is not None else None,
            )

        else:   # ip-adapter / instant-id
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
                        r = min(image.shape[0] // 224, image.shape[1] // 224)
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

            unet = IPAdapterApply().apply_ipadapter(
                ipadapter=self.ip_adapter,
                model=unet,
                weight=self.strength,
                start_at=self.start_percent,
                end_at=self.end_percent,
                faceid_v2=self.faceid_v2,
                weight_v2=self.strength*self.weight_v2,
                attn_mask=mask.squeeze(1) if mask is not None else None,
                **pcond,
            )

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(IPAdapterPatcher)

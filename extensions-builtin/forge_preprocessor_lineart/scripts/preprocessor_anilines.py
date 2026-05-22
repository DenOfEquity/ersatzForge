from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules import devices
from modules_forge.utils import resize_image_with_pad, HWC3
from modules.modelloader import load_file_from_url

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy
from PIL import Image, ImageEnhance


# derived from https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor/blob/master/infer.py

class LineExtractor(nn.Module):
    def __init__(self, chan_in, chan_out, bilinear=False):
        super().__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(chan_in, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, chan_out))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PreprocessorAniLines(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['Lineart']
        self.model_filename_filters = ['lineart']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.model = None
        self.device = devices.get_device_for('controlnet')
        
        self.cache = None
        self.cacheHash = None

    def load_model(self, name):
        if name == 'basic.pth':
            model = LineExtractor(3, 1, True)
        elif name == 'detail.pth':
            model = LineExtractor(2, 1, True)
        else:
            return None

        remote_model_path = 'https://huggingface.co/gyrojeff/AniLines/resolve/main/' + name
        model_dir = os.path.join(preprocessor_dir, 'AniLines')
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        return model

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):

        match self.name:
            case 'AniLines basic':
                if self.model is None:
                    self.model = self.load_model("basic.pth")
            case 'AniLines detail':
                if self.model is None:
                    self.model = self.load_model("detail.pth")
            case _:
                return input_image

        if self.model is None:
            return input_image
        else:
            image, remove_pad = resize_image_with_pad(input_image, resolution)
            self.model.to(self.device)

            if self.name == 'AniLines basic':
                img = Image.fromarray(image)
                enhancer = ImageEnhance.Sharpness(img)
                img = numpy.array(enhancer.enhance(6.0))

                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            else:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            
                img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
                sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            
                img = torch.cat([img, sobel], dim=1)

            B, C, H, W = img.shape
            pad_h = 8 - (H % 8)
            pad_w = 8 - (W % 8)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                pred = self.model(img)
            pred = ((pred[:, :, :H, :W].clamp(min=0.0, max=1.0) * 255.0) + 0.5)
            
            result = pred[0, 0].cpu().numpy().astype(numpy.uint8)
            
            self.model.to('cpu')

            return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorAniLines('AniLines basic'))
add_supported_preprocessor(PreprocessorAniLines('AniLines detail'))

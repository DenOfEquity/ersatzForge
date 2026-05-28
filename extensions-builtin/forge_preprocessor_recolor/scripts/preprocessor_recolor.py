import cv2
import numpy

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor


class PreprocessorRecolor(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['Recolor']
        self.model_filename_filters = ['color', 'recolor', 'grey', 'gray']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(
            visible=True,
            label="Gamma Correction",
            value=1.0,
            minimum=0.3,
            maximum=3.0,
            step=0.01
        )
        self.cond_mean = None

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        gamma = slider_1

        if self.name == "recolor_intensity":
            result = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            result = result[:, :, 2].astype(numpy.float32) / 255.0
        else:
            result = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
            result = result[:, :, 0].astype(numpy.float32) / 255.0

        result = result ** (1.0/gamma)
        result = (result * 255.0).clip(0, 255).astype(numpy.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return result

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        self.cond_mean = cond[0].mean(dim=0, keepdim=True)
        return cond, mask

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        batch_result = args[0].images
        new_results = []

        for img in batch_result:
            img = img - img.mean(dim=0, keepdim=True) + self.cond_mean
            img = img.clip(0, 1)
            new_results.append(img)

        args[0].images = new_results
        return


add_supported_preprocessor(PreprocessorRecolor(name="recolor_intensity",))
add_supported_preprocessor(PreprocessorRecolor(name="recolor_luminance",))

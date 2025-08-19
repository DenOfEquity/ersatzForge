import torch


class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177]
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076]
        ]
        self.taesd_decoder_name = "taesdxl_decoder"


class SDXL_Playground_2_5(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.5
        self.latents_mean = torch.tensor([-1.6574, 1.886, -1.383, 2.5155]).view(1, 4, 1, 1)
        self.latents_std = torch.tensor([8.4927, 5.9022, 6.5498, 5.2299]).view(1, 4, 1, 1)

        self.latent_rgb_factors = [
            #   R        G        B
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076]
        ]
        self.taesd_decoder_name = "taesdxl_decoder"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class SD_X4(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.08333
        self.latent_rgb_factors = [
            [-0.2340, -0.3863, -0.3257],
            [0.0994, 0.0885, -0.0908],
            [-0.2833, -0.2349, -0.3741],
            [0.2523, -0.0055, -0.1651]
        ]


class SC_Prior(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.0326, -0.0204, -0.0127],
            [-0.1592, -0.0427, 0.0216],
            [0.0873, 0.0638, -0.0020],
            [-0.0602, 0.0442, 0.1304],
            [0.0800, -0.0313, -0.1796],
            [-0.0810, -0.0638, -0.1581],
            [0.1791, 0.1180, 0.0967],
            [0.0740, 0.1416, 0.0432],
            [-0.1745, -0.1888, -0.1373],
            [0.2412, 0.1577, 0.0928],
            [0.1908, 0.0998, 0.0682],
            [0.0209, 0.0365, -0.0092],
            [0.0448, -0.0650, -0.1728],
            [-0.1658, -0.1045, -0.1308],
            [0.0542, 0.1545, 0.1325],
            [-0.0352, -0.1672, -0.2541]
        ]


class SC_B(LatentFormat):
    def __init__(self):
        self.scale_factor = 1.0 / 0.43
        self.latent_rgb_factors = [
            [0.1121, 0.2006, 0.1023],
            [-0.2093, -0.0222, -0.0195],
            [-0.3087, -0.1535, 0.0366],
            [0.0290, -0.1574, -0.4078]
        ]


class SD3(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0645, 0.0177, 0.1052],
            [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360],
            [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364],
            [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283],
            [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700],
            [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220],
            [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207],
            [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456],
            [-0.1418, -0.1457, -0.1259]
        ]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class StableAudio1(LatentFormat):
    latent_channels = 64


class Flux(SD3):
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0404, 0.0159, 0.0609],
            [0.0043, 0.0298, 0.0850],
            [0.0328, -0.0749, -0.0503],
            [-0.0245, 0.0085, 0.0549],
            [0.0966, 0.0894, 0.0530],
            [0.0035, 0.0399, 0.0123],
            [0.0583, 0.1184, 0.1262],
            [-0.0191, -0.0206, -0.0306],
            [-0.0324, 0.0055, 0.1001],
            [0.0955, 0.0659, -0.0545],
            [-0.0504, 0.0231, -0.0013],
            [0.0500, -0.0008, -0.0088],
            [0.0982, 0.0941, 0.0976],
            [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020],
            [-0.1273, -0.0932, -0.0680]
        ]

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class Wan21(LatentFormat):
    # latent_channels = 16
    # latent_dimensions = 3

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, 16, 1, 1, 1)
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, 16, 1, 1, 1)

        self.latent_rgb_factors = [
                [-0.1299, -0.1692,  0.2932],
                [ 0.0671,  0.0406,  0.0442],
                [ 0.3568,  0.2548,  0.1747],
                [ 0.0372,  0.2344,  0.1420],
                [ 0.0313,  0.0189, -0.0328],
                [ 0.0296, -0.0956, -0.0665],
                [-0.3477, -0.4059, -0.2925],
                [ 0.0166,  0.1902,  0.1975],
                [-0.0412,  0.0267, -0.1364],
                [-0.1293,  0.0740,  0.1636],
                [ 0.0680,  0.3019,  0.1128],
                [ 0.0032,  0.0581,  0.0639],
                [-0.1251,  0.0927,  0.1699],
                [ 0.0060, -0.0633,  0.0005],
                [ 0.3477,  0.2275,  0.2950],
                [ 0.1984,  0.0913,  0.1861]
            ]

        self.latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

        self.taesd_decoder_name = None

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean

class Wan22(Wan21):
    latent_channels = 48
    latent_dimensions = 3

    latent_rgb_factors = [
        [ 0.0119,  0.0103,  0.0046],
        [-0.1062, -0.0504,  0.0165],
        [ 0.0140,  0.0409,  0.0491],
        [-0.0813, -0.0677,  0.0607],
        [ 0.0656,  0.0851,  0.0808],
        [ 0.0264,  0.0463,  0.0912],
        [ 0.0295,  0.0326,  0.0590],
        [-0.0244, -0.0270,  0.0025],
        [ 0.0443, -0.0102,  0.0288],
        [-0.0465, -0.0090, -0.0205],
        [ 0.0359,  0.0236,  0.0082],
        [-0.0776,  0.0854,  0.1048],
        [ 0.0564,  0.0264,  0.0561],
        [ 0.0006,  0.0594,  0.0418],
        [-0.0319, -0.0542, -0.0637],
        [-0.0268,  0.0024,  0.0260],
        [ 0.0539,  0.0265,  0.0358],
        [-0.0359, -0.0312, -0.0287],
        [-0.0285, -0.1032, -0.1237],
        [ 0.1041,  0.0537,  0.0622],
        [-0.0086, -0.0374, -0.0051],
        [ 0.0390,  0.0670,  0.2863],
        [ 0.0069,  0.0144,  0.0082],
        [ 0.0006, -0.0167,  0.0079],
        [ 0.0313, -0.0574, -0.0232],
        [-0.1454, -0.0902, -0.0481],
        [ 0.0714,  0.0827,  0.0447],
        [-0.0304, -0.0574, -0.0196],
        [ 0.0401,  0.0384,  0.0204],
        [-0.0758, -0.0297, -0.0014],
        [ 0.0568,  0.1307,  0.1372],
        [-0.0055, -0.0310, -0.0380],
        [ 0.0239, -0.0305,  0.0325],
        [-0.0663, -0.0673, -0.0140],
        [-0.0416, -0.0047, -0.0023],
        [ 0.0166,  0.0112, -0.0093],
        [-0.0211,  0.0011,  0.0331],
        [ 0.1833,  0.1466,  0.2250],
        [-0.0368,  0.0370,  0.0295],
        [-0.3441, -0.3543, -0.2008],
        [-0.0479, -0.0489, -0.0420],
        [-0.0660, -0.0153,  0.0800],
        [-0.0101,  0.0068,  0.0156],
        [-0.0690, -0.0452, -0.0927],
        [-0.0145,  0.0041,  0.0015],
        [ 0.0421,  0.0451,  0.0373],
        [ 0.0504, -0.0483, -0.0356],
        [-0.0837,  0.0168,  0.0055],
    ]

    latent_rgb_factors_bias = [0.0317, -0.0878, -0.1388]

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor(
            [
                -0.2289,                -0.0052,                -0.1323,                -0.2339,
                -0.2799,                 0.0174,                 0.1838,                 0.1557,
                -0.1382,                 0.0542,                 0.2813,                 0.0891,
                 0.1570,                -0.0098,                 0.0375,                -0.1825,
                -0.2246,                -0.1207,                -0.0698,                 0.5109,
                 0.2665,                -0.2108,                -0.2158,                 0.2502,
                -0.2055,                -0.0322,                 0.1109,                 0.1567,
                -0.0729,                 0.0899,                -0.2799,                -0.1230,
                -0.0313,                -0.1649,                 0.0117,                 0.0723,
                -0.2839,                -0.2083,                -0.0520,                 0.3748,
                 0.0152,                 0.1957,                 0.1433,                -0.2944,
                 0.3573,                -0.0548,                -0.1681,                -0.0667,
            ]
        ).view(1, self.latent_channels, 1, 1)
        self.latents_std = torch.tensor([0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]).view(1, self.latent_channels, 1, 1)

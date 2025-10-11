import dataclasses
import torch
import k_diffusion
import numpy as np
from scipy import stats

from modules import shared
from backend.modules.k_prediction import Prediction


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: any

    default_rho: float = -1
    need_inner_model: bool = False
    aliases: list = None


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [
        inner_model.t_to_sigma(ts)
        for ts in torch.linspace(start, end, n + 1)[:-1]
    ]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]

    if isinstance(inner_model.predictor, Prediction):
        lo = sigs[-1]
        hi = sigs[0] - lo
        for x in range(n):
            sigs[x] -= lo
            sigs[x] /= hi
            sigs[x] *= (sigma_max-sigma_min)
            sigs[x] += sigma_min

    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]

    if isinstance(inner_model.predictor, Prediction):
        lo = sigs[-1]
        hi = sigs[0] - lo
        for x in range(n):
            sigs[x] -= lo
            sigs[x] /= hi
            sigs[x] *= (sigma_max-sigma_min)
            sigs[x] += sigma_min

    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    # From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024) """

    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta

    total_timesteps = (len(inner_model.sigmas) - 1)
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.ceil(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(inner_model.sigmas[int(t)])]
        last_t = t

    if isinstance(inner_model.predictor, Prediction):
        lo = sigs[-1]
        hi = sigs[0] - lo
        for x in range(n):
            sigs[x] -= lo
            sigs[x] /= hi
            sigs[x] *= (sigma_max-sigma_min)
            sigs[x] += sigma_min

    sigs += [0.0]

    return torch.FloatTensor(sigs).to(device)


def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().to(torch.int64).clip(0, 999)
    sigmas = unet.model.predictor.sigma(timesteps)
    
    if isinstance(inner_model.predictor, Prediction):
        lo = sigmas[-1].clone()
        hi = sigmas[0].clone() - lo

        sigmas -= lo
        sigmas /= hi
        sigmas *= (sigma_max-sigma_min)
        sigmas += sigma_min
    
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def get_align_your_steps_sigmas_GITS(n, sigma_min, sigma_max, device):
    if shared.sd_model.is_sdxl:
        sigmas = [14.615, 4.734, 2.567, 1.529, 0.987, 0.652, 0.418, 0.268, 0.179, 0.127, 0.029]
    else:
        sigmas = [14.615, 4.617, 2.507, 1.236, 0.702, 0.402, 0.240, 0.156, 0.104, 0.094, 0.029]

    for x in range(len(sigmas)):
        sigmas[x] -= 0.029
        sigmas[x] /= 14.615 - 0.029
        sigmas[x] *= (sigma_max-sigma_min)
        sigmas[x] += sigma_min

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def ays_11_sigmas(n, sigma_min, sigma_max, device='cpu'):
    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html

    if shared.sd_model.is_sdxl:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    for x in range(len(sigmas)):
        sigmas[x] -= 0.029
        sigmas[x] /= 14.615 - 0.029
        sigmas[x] *= (sigma_max-sigma_min)
        sigmas[x] += sigma_min

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def ays_32_sigmas(n, sigma_min, sigma_max, device='cpu'):
    if shared.sd_model.is_sdxl:
        sigmas = [14.61500000000000000, 11.14916180000000000, 8.505221270000000000, 6.488271510000000000, 5.437074020000000000, 4.603986190000000000, 3.898547040000000000, 3.274074570000000000, 2.743965270000000000, 2.299686590000000000, 1.954485140000000000, 1.671087150000000000, 1.428781520000000000, 1.231810090000000000, 1.067896490000000000, 0.925794430000000000, 0.802908860000000000, 0.696601210000000000, 0.604369030000000000, 0.528525520000000000, 0.467733440000000000, 0.413933790000000000, 0.362581860000000000, 0.310085170000000000, 0.265189250000000000, 0.223264610000000000, 0.176538770000000000, 0.139591920000000000, 0.105873810000000000, 0.055193690000000000, 0.028773340000000000, 0.015000000000000000]
    else:
        sigmas = [14.61500000000000000, 11.23951352000000000, 8.643630810000000000, 6.647294240000000000, 5.572508620000000000, 4.716485460000000000, 3.991960650000000000, 3.519560900000000000, 3.134904660000000000, 2.792287880000000000, 2.487736280000000000, 2.216638650000000000, 1.975083510000000000, 1.779317200000000000, 1.614753350000000000, 1.465409530000000000, 1.314849000000000000, 1.166424970000000000, 1.034755470000000000, 0.915737440000000000, 0.807481690000000000, 0.712023610000000000, 0.621739000000000000, 0.530652020000000000, 0.452909600000000000, 0.374914550000000000, 0.274618190000000000, 0.201152900000000000, 0.141058730000000000, 0.066828810000000000, 0.031661210000000000, 0.015000000000000000]

    for x in range(len(sigmas)):
        sigmas[x] -= 0.015
        sigmas[x] /= 14.615 - 0.015
        sigmas[x] *= (sigma_max-sigma_min)
        sigmas[x] += sigma_min

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)
    return torch.FloatTensor(sigmas).to(device)


def sigmoid_offset_sigmas(n, sigma_min, sigma_max, inner_model, device):
    square_k = shared.opts.sigmoid_square_k
    base_c = shared.opts.sigmoid_base_c
    total_timesteps = len(inner_model.sigmas)-1
    ts = np.linspace(0, 1, n, endpoint=False)
    shift = 2.0 * (base_c - 0.5)
    def sigmoid(x):
        x = (8.0 * x - 4.0) + (shift * 4.0)
        if square_k * x > 700:
            return 1.0
        if square_k * x < -700:
            return 0.0
        return 1.0 / (1.0 + np.exp(-square_k * x))
    transformed_ts = 1.0 - np.array([sigmoid(t) for t in ts])

    # range fix
    maximum = transformed_ts.max()
    transformed_ts /= maximum

    mapped_ts = np.rint(transformed_ts * total_timesteps).astype(int)
    sigs = []
    for t in mapped_ts:
        sigs.append(float(inner_model.sigmas[t]))

    if isinstance(inner_model.predictor, Prediction):
        lo = sigs[-1]
        hi = sigs[0] - lo
        for x in range(n):
            sigs[x] -= lo
            sigs[x] /= hi
            sigs[x] *= (sigma_max-sigma_min)
            sigs[x] += sigma_min

    sigs.append(0.0)
    return torch.FloatTensor(sigs).to(device)


# bong_tangent from RES4LYF nodes by ClownsharkBatwing
import math
def bong_tangent_scheduler(n, sigma_min, sigma_max, inner_model, device):
    def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
        smax = 0.5 * ((2 / math.pi) * math.atan(-slope * (0-pivot)) + 1)
        smin = 0.5 * ((2 / math.pi) * math.atan(-slope * ((steps-1)-pivot)) + 1)

        srange = smax - smin
        sscale = start - end

        sigmas = [((0.5 * ((2 / math.pi) * math.atan(-slope * (x-pivot))+1)) - smin) * (1 / srange) * sscale + end    for x in range(steps)]
        
        return sigmas

    start = 1.0
    middle = 0.5
    end = 0.0
    pivot_1 = 0.6
    pivot_2 = 0.6
    slope_1 = 0.2
    slope_2 = 0.2
    pad = True

    if pad:
        n += 1
    else:
        n += 2

    midpoint = int( (n*pivot_1 + n*pivot_2) / 2 )
    pivot_1 = int(n * pivot_1)
    pivot_2 = int(n * pivot_2)

    slope_1 = slope_1 / (n/40)
    slope_2 = slope_2 / (n/40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
    
    tan_sigmas_1 = tan_sigmas_1[:-1]

    tan_sigmas = tan_sigmas_1 + tan_sigmas_2

    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(len(tan_sigmas)):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]

    if isinstance(inner_model.predictor, Prediction):
        lo = sigs[-1]
        hi = sigs[0] - lo
        for x in range(len(sigs)):
            sigs[x] -= lo
            sigs[x] /= hi
            sigs[x] *= (sigma_max-sigma_min)
            sigs[x] += sigma_min

    if pad:
        sigs += [0.0]

    return torch.FloatTensor(sigs).to(device)


schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
    Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
    Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
    Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
    Scheduler('turbo', 'Turbo', turbo_scheduler, need_inner_model=True),
    Scheduler('align_your_steps_GITS', 'Align Your Steps GITS', get_align_your_steps_sigmas_GITS),
    Scheduler('align_your_steps_11', 'Align Your Steps 11', ays_11_sigmas),
    Scheduler('align_your_steps_32', 'Align Your Steps 32', ays_32_sigmas),
    Scheduler('sigmoid_offset', 'Sigmoid Offset', sigmoid_offset_sigmas, need_inner_model=True),
    Scheduler('bong_tangent', 'Bong Tangent', bong_tangent_scheduler, need_inner_model=True),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}

def visible_scheduler_names():
    global schedulers
    return [x.label for x in schedulers if x.label not in shared.opts.hide_schedulers]

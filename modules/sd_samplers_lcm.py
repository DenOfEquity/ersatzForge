import torch

from k_diffusion import utils, sampling
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import default_noise_sampler, trange

from modules import shared, sd_samplers_cfg_denoiser, sd_samplers_kdiffusion, sd_samplers_common


class LCMCompVisDenoiser(DiscreteEpsDDPMDenoiser):
    def __init__(self, model):
        timesteps = 1000
        original_timesteps = 50     # LCM Original Timesteps (default=50, for current version of LCM)
        self.skip_steps = timesteps // original_timesteps

        alphas_cumprod = 1.0 / (model.forge_objects.unet.model.predictor.sigmas ** 2.0 + 1.0)
        alphas_cumprod_valid = torch.zeros(original_timesteps, dtype=torch.float32)
        for x in range(original_timesteps):
            alphas_cumprod_valid[original_timesteps - 1 - x] = alphas_cumprod[timesteps - 1 - x * self.skip_steps]

        super().__init__(model, alphas_cumprod_valid, quantize=None)
        self.predictor = model.forge_objects.unet.model.predictor


def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    order = shared.opts.lcm_order

    previous1 = None
    previous2 = None
    previous3 = None
    previous4 = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised

        if previous4 is not None:
            previous = 0.0625 * (9*previous1 + 4*previous2 + 2*previous3 + previous4)
        elif previous3 is not None:
            previous = 0.125  * (5*previous1 + 2*previous2 + previous3)
        elif previous2 is not None:
            previous = 0.25   * (3*previous1 + previous2)
        elif previous1 is not None:
            previous = previous1
        else:
            previous = None

        if previous is not None:
            x += (denoised - previous) * (sigmas[i] / sigmas[0])

        if order == 5: previous4 = previous3
        if order >= 4: previous3 = previous2
        if order >= 3: previous2 = previous1
        if order >= 2: previous1 = x

        if sigmas[i + 1] > 0:
            x.addcmul_(sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]))
    return x


class CFGDenoiserLCM(sd_samplers_cfg_denoiser.CFGDenoiser):
    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = LCMCompVisDenoiser
            self.model_wrap = denoiser(shared.sd_model)

        return self.model_wrap


class LCMSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname, sd_model, options)
        self.model_wrap_cfg = CFGDenoiserLCM(self)
        self.model_wrap = self.model_wrap_cfg.inner_model


samplers_lcm = [('LCM', sample_lcm, ['k_lcm'], {}), ]
samplers_data_lcm = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: LCMSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_lcm
]

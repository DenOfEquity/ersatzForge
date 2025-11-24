import torch

from k_diffusion.sampling import default_noise_sampler, trange
from modules import shared, sd_samplers_kdiffusion, sd_samplers_common


@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    order = shared.opts.lcm_order
    scale = shared.opts.lcm_noise

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

        if order == 5 and i > 3: previous4 = previous3.clone()
        if order >= 4 and i > 2: previous3 = previous2.clone()
        if order >= 3 and i > 1: previous2 = previous1.clone()
        if order >= 2 and i > 0: previous1 = x.clone()

        if sigmas[i + 1] > 0:
            noise_scaling = sigmas[i + 1]
            if scale < 1.0 and noise_scaling.item() > 1.0:
                noise_scaling **= scale
            x = model.inner_model.predictor.noise_scaling(noise_scaling, noise_sampler(sigmas[i], sigmas[i + 1]), x)
    return x


samplers_lcm = [('LCM', sample_lcm, ['k_lcm'], {}), ]
samplers_data_lcm = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_lcm
]

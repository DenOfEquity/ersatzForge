from modules import sd_samplers_kdiffusion, sd_samplers_common
from backend.modules import k_diffusion_extra


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name):
        sampler_function = getattr(k_diffusion_extra, sampler_name)
        super().__init__(sampler_function, sd_model, None)


def build_constructor(sampler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='sample_ddpm'), ['ddpm'], {}),
    sd_samplers_common.SamplerData('SA-Solver', build_constructor(sampler_name='sample_sa_solver'), ['sa_solver'], {}),
    sd_samplers_common.SamplerData('Extended Reverse-Time SDE', build_constructor(sampler_name='sample_er_sde'), ['er_sde'], {}),
    sd_samplers_common.SamplerData('Adaptive-ODE', build_constructor(sampler_name='sample_adaptive_ode'), ['adaptive_ode'], {}),
    sd_samplers_common.SamplerData('Fixed-ODE', build_constructor(sampler_name='sample_fixed_ode'), ['fixed_ode'], {}),
]
sd_samplers_kdiffusion.sampler_extra_params.update({
    'sample_sa_solver': ['s_noise'],
    'sample_er_sde'   : ['s_noise'],
})

# from __future__ import annotations

# import functools
# import logging
from modules import sd_samplers_kdiffusion, sd_samplers_timesteps, sd_samplers_lcm, shared, sd_samplers_common, sd_schedulers

# imports for functions that previously were here and are used by other modules
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image  # noqa: F401
from modules_forge import alter_samplers

all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
    *sd_samplers_lcm.samplers_data_lcm,
    *alter_samplers.samplers_data_alter
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers: list[sd_samplers_common.SamplerData] = []
samplers_for_img2img: list[sd_samplers_common.SamplerData] = []
samplers_map = {}


def find_sampler_config(name):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    return config


def create_sampler(name, model):
    config = find_sampler_config(name)

    assert config is not None, f'bad sampler name: {name}'

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers():
    global samplers, samplers_for_img2img

    samplers = all_samplers
    samplers_for_img2img = all_samplers

    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name

    return


def add_sampler(sampler):
    global all_samplers, all_samplers_map
    if sampler.name not in [x.name for x in all_samplers]:
        all_samplers.append(sampler)
        all_samplers_map = {x.name: x for x in all_samplers}
        set_samplers()
    return


def visible_sampler_names():
    return [x.name for x in samplers if x.name not in shared.opts.hide_samplers]


def fix_p_invalid_sampler_and_scheduler(p):
    sampler_names = [x.name for x in samplers]
    scheduler_names = [x.label for x in sd_schedulers.schedulers]

    # first check for old form, combined sampler name + scheduler name
    if p.sampler_name not in sampler_names:
        for scheduler in scheduler_names:
            if p.sampler_name.endswith(" " + scheduler):
                p.sampler_name = p.sampler_name[0:-(len(scheduler) + 1)]
                p.scheduler = scheduler
                print (f"[Autocorrection] Sampler: {p.sampler_name}; Scheduler: {p.scheduler}.")
                break

    if p.sampler_name not in sampler_names:
        print (f"[Autocorrection] Unknown sampler {p.sampler_name}, defaulted to {samplers[0].name}.")
        p.sampler_name = samplers[0].name

    if p.scheduler not in scheduler_names:
        print (f"[Autocorrection] Unknown scheduler {p.scheduler}, defaulted to {sd_schedulers.schedulers[0].label}.")
        p.scheduler = sd_schedulers.schedulers[0].label



set_samplers()

import torch
import math

from modules import devices, rng_philox, shared

#### perlin noise via Extraltodeus. modified for noise shape; to use Generator for batch consistency
#    found at https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
#    which was ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
def rand_perlin_2d(shape, generator, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1, generator=generator)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, generator, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, generator, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    noise = torch.remainder(torch.abs(noise)*1000000,11)/11
    return noise

# input width/height is latent width/height, not image
def create_noisy_latents_perlin(seed, shape, generator=None):
    detail_level = 1    # Setting?
    channels, height, width = shape
    if generator is None and seed != -1:
        torch.manual_seed(seed)
    noise = torch.zeros(shape, dtype=torch.float32, device="cpu")
    for j in range(channels):
        noise_values = rand_perlin_2d_octaves((height, width), generator, (1,1), 1, 1)
        result = (1+detail_level/10)*torch.erfinv(2 * noise_values - 1) * (2 ** 0.5)
        result = torch.clamp(result,-5,5)
        noise[j, :, :] = result

    return noise
#### end: perlin


def get_noise_source_type():
    if shared.opts.forge_try_reproduce in ['ComfyUI', 'DrawThings']:
        return "CPU"

    return shared.opts.randn_source


def randn(seed, shape, generator=None):
    """Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed."""

    if generator is None:
        manual_seed(seed)

    match get_noise_source_type():
        case "Perlin":
            return create_noisy_latents_perlin(seed, shape, generator).to(devices.device)
        case "NV":
            return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)
        case "CPU":
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        case _:
            if devices.device.type == 'mps':
                return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
            return torch.randn(shape, device=devices.device, generator=generator)


def randn_local(seed, shape):
    """Does not change the global random number generator. You can only generate the seed's first tensor using this function."""

    match get_noise_source_type():
        case "Perlin":
            return create_noisy_latents_perlin(seed, shape).to(devices.device)
        case "NV":
            rng = rng_philox.Generator(seed)
            return torch.asarray(rng.randn(shape), device=devices.device)
        case "CPU":
            local_generator = torch.Generator(devices.cpu).manual_seed(int(seed))
            return torch.randn(shape, device=devices.cpu, generator=local_generator).to(devices.device)
        case _:
            local_device = devices.cpu if devices.device.type == 'mps' else devices.device
            local_generator = torch.Generator(local_device).manual_seed(int(seed))
            return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)


def randn_like(x):
    """Use either randn() or manual_seed() to initialize the generator."""

    match get_noise_source_type():
        case "Perlin": # check this with ancestral
            return create_noisy_latents_perlin(-1, x.shape).to(x.device)
        case "NV":
            return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)
        case "CPU":
            return torch.randn_like(x, device=devices.cpu).to(x.device)
        case _:
            if x.device.type == 'mps':
                return torch.randn_like(x, device=devices.cpu).to(x.device)
            return torch.randn_like(x)


def randn_without_seed(shape, generator=None):
    """Use either randn() or manual_seed() to initialize the generator."""

    match get_noise_source_type():
        case "Perlin":
            return create_noisy_latents_perlin(-1, shape, generator).to(devices.device)
        case "NV":
            return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)
        case "CPU":
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        case _:
            if devices.device.type == 'mps':
                return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
            return torch.randn(shape, device=devices.device, generator=generator)


def manual_seed(seed):
    """Set up a global random number generator using the specified seed."""

    if get_noise_source_type() == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


def create_generator(seed):
    match get_noise_source_type():
        case "Perlin":
            return torch.Generator(devices.cpu).manual_seed(int(seed))
        case "NV":
            return rng_philox.Generator(seed)
        case "CPU":
            return torch.Generator(devices.cpu).manual_seed(int(seed))
        case _:
            device = devices.cpu if devices.device.type == 'mps' else devices.device
            return torch.Generator(device).manual_seed(int(seed))


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


class ImageRNG:
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        self.generators = [create_generator(seed) for seed in seeds]

        self.is_first = True

    def first(self):
        noise_shape = self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 else (self.shape[0], int(self.seed_resize_from_h) // 8, int(self.seed_resize_from_w // 8))

        xs = []

        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(subseed, noise_shape)

            if noise_shape != self.shape:
                noise = randn(seed, noise_shape)
            else:
                noise = randn(seed, self.shape, generator=generator)

            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            if noise_shape != self.shape:
                x = randn(seed, self.shape, generator=generator)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
                noise = x

            xs.append(noise)

        eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
        if eta_noise_seed_delta:
            self.generators = [create_generator(seed + eta_noise_seed_delta) for seed in self.seeds]

        return torch.stack(xs).to(shared.device)

    def next(self):
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = []
        for generator in self.generators:
            x = randn_without_seed(self.shape, generator=generator)
            xs.append(x)

        return torch.stack(xs).to(shared.device)


devices.randn = randn
devices.randn_local = randn_local
devices.randn_like = randn_like
devices.randn_without_seed = randn_without_seed
devices.manual_seed = manual_seed

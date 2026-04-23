import torch

from modules import devices, rng_philox, shared

#### perlin noise via Extraltodeus.
#    found at https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
#    which was ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
##   DoE modified for: noise shape; to use Generator for batch consistency; fix error at [0,0]; erfinv scaling to avoid -Inf; Settings; GPU+CPU compatible
##       vectorization courtesy Copilot
def rand_perlin_2d(shape, generator, frequency):
    device = generator.device
    pi = 3.141593
    sqrt2 = 1.414214
    delta = (frequency / shape[0], frequency / shape[1])
    d = (shape[0] // frequency, shape[1] // frequency)

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, frequency, delta[0], device=device, dtype=torch.float32),
            torch.arange(0, frequency, delta[1], device=device, dtype=torch.float32)
        ), dim=-1
    ) % 1
    angles = 2*pi*torch.rand(frequency+1, frequency+1, generator=generator, device=device, dtype=torch.float32)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    # Base grid
    base = torch.stack(
        (grid[:shape[0], :shape[1], 0],
         grid[:shape[0], :shape[1], 1]), dim=-1
    )

    # Offsets for the 4 corners
    offsets = torch.tensor([
        [ 0,  0],   # n00
        [-1,  0],   # n10
        [ 0, -1],   # n01
        [-1, -1],   # n11
    ], device=base.device, dtype=base.dtype)

    # Apply offsets
    grid4 = base[None, ...] + offsets[:, None, None, :]

    # Gradient fields
    g00 = gradients[0:-1, 0:-1]
    g10 = gradients[1:  , 0:-1]
    g01 = gradients[0:-1, 1:  ]
    g11 = gradients[1:  , 1:  ]

    grads4 = torch.stack((g00, g10, g01, g11), dim=0)
    grads4 = grads4.repeat_interleave(d[0], 1).repeat_interleave(d[1], 2)
    grads4 = grads4[:, :shape[0], :shape[1]]

    # Vectorized dot products
    dots = (grid4 * grads4).sum(dim=-1)

    n00, n10, n01, n11 = dots

    fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3
    t = fade(grid[:shape[0], :shape[1]])
    u = t[..., 0]
    v = t[..., 1]
    nx0 = torch.lerp(n00, n10, u)
    nx1 = torch.lerp(n01, n11, u)
    nxy = torch.lerp(nx0, nx1, v)

    return sqrt2 * nxy


def rand_perlin_2d_octaves(shape, generator, octaves=1, persistence=0.5):
    noise = torch.zeros(shape, dtype=torch.float32, device=generator.device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, generator, frequency)
        frequency *= 2
        amplitude *= persistence
    noise = torch.remainder(torch.abs(noise)*1000000,11)/11
    return noise

# input width/height is latent width/height, not image
def create_noisy_latents_perlin(seed, shape, generator=None):
    detail_level = shared.opts.perlin_detail
    octaves = shared.opts.perlin_octaves    # must scale into actual latent size
    power2 = [1, 2, 4, 8, 16, 32, 64, 128]  # seven octaves max, arbitrarily
    while shape[-1] % power2[octaves-1] != 0:
        octaves -= 1
    while shape[-2] % power2[octaves-1] != 0:
        octaves -= 1
    persistence = shared.opts.perlin_persist
    channels, height, width = shape
    if generator is None and seed != -1:
        torch.manual_seed(seed)
    noise = torch.empty((channels, height, width), dtype=torch.float32, device=generator.device)
    for j in range(channels):
        noise_values = rand_perlin_2d_octaves((height, width), generator, octaves, persistence)
        noise_values[:1, :1] = noise_values[-1:, -1:]   # fix bad result at [0,0]
        result = (1+detail_level/10)*torch.erfinv(1.9992 * noise_values - 0.9996) * (2 ** 0.5) # erfinv input range should be (-1, 1)
        # result.clamp_(result,-5,5)
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
        case "Perlin (GPU)":
            return create_noisy_latents_perlin(seed, shape, generator).to(devices.device)
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
        case "Perlin (GPU)":
            local_device = devices.cpu if devices.device.type == 'mps' else devices.device
            local_generator = torch.Generator(local_device).manual_seed(int(seed))
            return create_noisy_latents_perlin(seed, shape, local_generator).to(devices.device)
        case "Perlin":
            local_generator = torch.Generator(devices.cpu).manual_seed(int(seed))
            return create_noisy_latents_perlin(seed, shape, local_generator).to(devices.device)
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
        case "Perlin (GPU)":
            return create_noisy_latents_perlin(-1, x.shape).to(x.device)
        case "Perlin":
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
        case "Perlin (GPU)":
            return create_noisy_latents_perlin(-1, shape, generator).to(devices.device)
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
        case "Perlin (GPU)":
            device = devices.cpu if devices.device.type == 'mps' else devices.device
            return torch.Generator(device).manual_seed(int(seed))
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

# Only include samplers that are not already in A1111

import torch
import math

from tqdm import tqdm, trange

from k_diffusion.sampling import default_noise_sampler
from modules import shared


def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)





def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)."""
    if hasattr(model_sampling, 'prediction_type') and model_sampling.prediction_type == 'const':
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()

def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR."""
    if len(sigmas) <= 1:
        return sigmas
    if hasattr(model_sampling, 'prediction_type') and model_sampling.prediction_type == 'const':
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas



##### SA-Solver: Stochastic Adams Solver (NeurIPS 2023, arXiv:2309.05019)
# Conference: https://proceedings.neurips.cc/paper_files/paper/2023/file/f4a6806490d31216a3ba667eb240c897-Paper-Conference.pdf
# Codebase ref: https://github.com/scxue/SA-Solver

def compute_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float) -> torch.Tensor:
    """Compute (1 + tau^2) * integral of exp((1 + tau^2) * x) * x^p dx from s to t with exp((1 + tau^2) * t) factored out, using integration by parts.

    Integral of exp((1 + tau^2) * x) * x^p dx
        = product_terms[p] - (p / (1 + tau^2)) * integral of exp((1 + tau^2) * x) * x^(p-1) dx,
    with base case p=0 where integral equals product_terms[0].

    where
        product_terms[p] = x^p * exp((1 + tau^2) * x) / (1 + tau^2).

    Construct a recursive coefficient matrix following the above recursive relation to compute all integral terms up to p = (solver_order - 1).
    Return coefficients used by the SA-Solver in data prediction mode.

    Args:
        s: Start time s.
        t: End time t.
        solver_order: Current order of the solver.
        tau_t: Stochastic strength parameter in the SDE.

    Returns:
        Exponential coefficients used in data prediction, with exp((1 + tau^2) * t) factored out, ordered from p=0 to p=solver_order−1, shape (solver_order,).
    """
    tau_mul = 1 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)

    # product_terms after factoring out exp((1 + tau^2) * t)
    # Includes (1 + tau^2) factor from outside the integral
    product_terms_factored = (t ** p - s ** p * (-tau_mul * h).exp())

    # Lower triangular recursive coefficient matrix
    # Accumulates recursive coefficients based on p / (1 + tau^2)
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()

    return recursive_coeff_mat @ product_terms_factored


def compute_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, simple_order_2: bool = False, is_corrector_step: bool = False) -> torch.Tensor:
    """Compute b_i coefficients for the SA-Solver (see eqs. 15 and 18).

    The solver order corresponds to the number of input lambdas (half-logSNR points).

    Args:
        sigma_next: Sigma at end time t.
        curr_lambdas: Lambda time points used to construct the Lagrange basis, shape (N,).
        lambda_s: Lambda at start time s.
        lambda_t: Lambda at end time t.
        tau_t: Stochastic strength parameter in the SDE.
        simple_order_2: Whether to enable the simple order-2 scheme.
        is_corrector_step: Flag for corrector step in simple order-2 mode.

    Returns:
        b_i coefficients for the SA-Solver, shape (N,), where N is the solver order.
    """
    num_timesteps = curr_lambdas.shape[0]

    if simple_order_2 and num_timesteps == 2:
        """Compute simple order-2 b coefficients from SA-Solver paper (Appendix D. Implementation Details)."""
        tau_mul = 1 + tau_t ** 2
        h = lambda_t - lambda_s
        alpha_t = sigma_next * lambda_t.exp()
        if is_corrector_step:
            # Simplified 1-step (order-2) corrector
            b_1 = alpha_t * (0.5 * tau_mul * h)
            b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
        else:
            # Simplified 2-step predictor
            b_2 = alpha_t * (0.5 * tau_mul * h ** 2) / (curr_lambdas[-2] - lambda_s)
            b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
        return torch.stack([b_2, b_1])

    # Compute coefficients by solving a linear system from Lagrange basis interpolation
    exp_integral_coeffs = compute_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_T = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_T, exp_integral_coeffs)

    # (sigma_t * exp(-tau^2 * lambda_t)) * exp((1 + tau^2) * lambda_t)
    # = sigma_t * exp(lambda_t) = alpha_t
    # exp((1 + tau^2) * lambda_t) is extracted from the integral
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


@torch.no_grad()
def sample_sa_solver(model, x, sigmas, extra_args=None, callback=None, disable=False, s_noise=1.0, noise_sampler=None):
    """Stochastic Adams Solver with predictor-corrector method (NeurIPS 2023)."""
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    predictor_order = 3
    corrector_order = 4
    simple_order_2 = False
    use_pece = True         # Predict–Evaluate–Correct–Evaluate

    model_sampling = model.inner_model.predictor
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    # Use default interval for stochastic sampling
    start_sigma = model_sampling.percent_to_sigma(0.2)
    end_sigma = model_sampling.percent_to_sigma(0.8)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x  # x: current state, x_pred: predicted next state

    h = 0.0
    tau_t = 0.0
    noise = 0.0
    pred_list = []

    # Lower order near the end to improve stability
    lower_order_to_end = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        # Evaluation
        denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x_pred, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector
        if corrector_order_used == 0:
            # Update by the predicted state
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)    # (B, K, ...)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                # The noise from the previous predictor step
                x = x + noise

            if use_pece:
                # Evaluate the corrected state
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                pred_list[-1] = denoised

        # Predictor
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            tau_t = 1.0 if start_sigma >= sigmas[i + 1] >= end_sigma else 0.0
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)    # (B, K, ...)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
    return x
#### end: SA Solver


#### ER-SDE
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None):
    """Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3). arXiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    def noise_scaler(x):
        return x * ((x ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)
    max_stage = 3

    model_sampling = model.inner_model.predictor
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda_t = sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt()#.nan_to_num(nan=0.0)
        old_denoised = denoised
    return x
#### end: ER-SDE


#### ODE-based sampling.
import torchdiffeq

class ODEFunction:
    def __init__(self, model, t_min, t_max, n_steps, is_adaptive, extra_args=None, callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.t_min = t_min.item()
        self.t_max = t_max.item()
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.step = 0

        if is_adaptive:
            self.pbar = tqdm(
                total=100,
                desc="solve",
                unit="%",
                leave=False,
                position=1
            )
        else:
            self.pbar = tqdm(
                total=n_steps,
                desc="solve",
                leave=False,
                position=1
            )

    def __call__(self, t, y):
        if t <= 1e-5:
            return torch.zeros_like(y)

        denoised = self.model(y.unsqueeze(0), t.unsqueeze(0), **self.extra_args)
        return (y - denoised.squeeze(0)) / t

    def _callback(self, t0, y0, step):
        if self.callback is not None:
            y0 = y0.unsqueeze(0)

            self.callback({
                "x": y0,
                "i": step,
                "sigma": t0,
                "sigma_hat": t0,
                "denoised": y0, # for a bad latent preview
            })

    def callback_step(self, t0, y0, dt):
        if self.is_adaptive:
            return

        self._callback(t0, y0, self.step)

        self.pbar.update(1)
        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        if not self.is_adaptive:
            return

        progress = (self.t_max - t0.item()) / (self.t_max - self.t_min)

        self._callback(t0, y0, round((self.n_steps - 1) * progress))

        new_step = round(100 * progress)
        self.pbar.update(new_step - self.step)
        self.step = new_step

    def reset(self):
        self.step = 0
        self.pbar.reset()

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams"}

@torch.no_grad()
def sample_adaptive_ode(model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
    t_max = sigmas.max()
    t_min = sigmas.min()
    n_steps = len(sigmas)
    
    solver = shared.opts.adaptive_ode_solver
    rtol = 10**shared.opts.adaptive_ode_rtol
    atol = 10**shared.opts.adaptive_ode_atol
    max_steps = 250

    if solver in FIXED_SOLVERS:
        t = sigmas
        is_adaptive = False
    else:
        t = torch.stack([t_max, t_min])
        is_adaptive = True

    ode = ODEFunction(model, t_min, t_max, n_steps, is_adaptive=is_adaptive, callback=callback, extra_args=extra_args)

    samples = torch.empty_like(x)
    for i in trange(x.shape[0], desc=solver, disable=disable):
        ode.reset()

        samples[i] = torchdiffeq.odeint(
            ode,
            x[i],
            t,
            rtol=rtol,
            atol=atol,
            method=solver,
            options={
                "min_step": 1e-5,
                "max_num_steps": max_steps,
                "dtype": torch.float32# if torch.backends.mps.is_available() else torch.float64
            }
        )[-1]

    if callback is not None:
        callback({
            "x": samples,
            "i": n_steps - 1,
            "sigma": t_min,
            "sigma_hat": t_min,
            "denoised": samples, # only accurate if t_min = 0, for now
        })

    return samples
#### end ODE

"""Flow Momentum Sampler - a custom sampler for rectified-flow models.

An experiment (started 2026-11-05): what happens if plain flow Euler
sampling carries MOMENTUM - an exponential moving average of the
velocity - instead of stepping along each step's raw velocity?

    v_t = (x_t - denoised_t) / sigma_t        (rectified-flow velocity)
    m_t = beta * m_{t-1} + (1 - beta) * v_t    (momentum buffer)
    x_{t+1} = x_t + m_t * (sigma_{t+1} - sigma_t)

beta = 0 degenerates to plain flow Euler. Larger beta smooths the
trajectory direction across steps; too large and the trajectory lags
where it curves (early steps) or overshoots.

Sigmas use the linear-shift grid of the official Anima Diffusers
scheduler: sigma = shift * t / (1 + (shift - 1) * t), t from denoise
down to 0. If an AuraFlow-style model patch is also applied, the
shift is applied twice - use one or the other.

The momentum is applied to the velocity (the flow-native quantity),
not to the denoised prediction. A two-term history method (AB2) is
the closest classical relative; this is its EMA generalisation.
"""

import torch

import comfy.sample
import comfy.samplers


def flow_linear_shift_sigmas(steps, shift, denoise):
    """Linear-shift sigma grid, descending, ending at exactly 0."""
    t = torch.linspace(float(denoise), 0.0, int(steps) + 1,
                       dtype=torch.float32)
    s = float(shift)
    sigmas = s * t / (1.0 + (s - 1.0) * t)
    sigmas[-1] = 0.0
    return sigmas


def _make_sampler_function(beta):
    beta = float(beta)

    def sampler_function(model, x, sigmas, extra_args=None,
                         callback=None, disable=None):
        extra_args = extra_args or {}
        momentum = None
        total = len(sigmas) - 1
        for i in range(total):
            sigma = sigmas[i]
            denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                             **extra_args)
            if float(sigma) == 0.0:
                x = denoised
            else:
                velocity = (x - denoised) / sigma
                if momentum is None:
                    momentum = velocity.clone()
                else:
                    momentum = beta * momentum + (1.0 - beta) * velocity
                x = x + momentum * (sigmas[i + 1] - sigma)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "denoised": denoised})
        return x

    return sampler_function


class FlowMomentumSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0,
                              "step": 0.1}),
            "momentum": ("FLOAT", {"default": 0.5, "min": 0.0,
                                   "max": 0.95, "step": 0.05}),
            "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                     "max": 10.0, "step": 0.1}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0,
                                  "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/flow"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, momentum, flow_shift, denoise):
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"])
        # Safety net: if the channel fix left a Cosmos/Anima latent
        # at 4D, add the temporal dim the DiT expects.
        _expand5d = (_model_wants_temporal(model)
                     and latent_samples.ndim == 4)
        if _expand5d:
            latent_samples = latent_samples.unsqueeze(2)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if _expand5d:
            noise = noise.unsqueeze(2)
        sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
        sampler = comfy.samplers.KSAMPLER(
            _make_sampler_function(momentum))
        print(f"[FlowMomentum] beta={momentum}, shift={flow_shift}, "
              f"steps={steps}, cfg={cfg}")
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        out = dict(latent_image)
        if _expand5d:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out,)


def _make_ab_sampler_function(order):
    """Variable-step Adams-Bashforth for rectified flow.

    order 1 = plain Euler. order 2 extrapolates the velocity from the
    previous step (coefficients adjusted for the local step-size ratio),
    which tracks curved trajectories at the same model-eval count.
    """

    def sampler_function(model, x, sigmas, extra_args=None,
                         callback=None, disable=None, **kwargs):
        extra_args = extra_args or {}
        sigmas = sigmas.to(x.device)
        v_prev = None
        d_prev = None
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                             **extra_args)
            d = sigmas[i + 1] - sigma
            if float(sigmas[i + 1]) == 0.0:
                x = denoised
            else:
                v = (x - denoised) / sigma
                if order >= 2 and v_prev is not None and float(d_prev) != 0.0:
                    r = d / d_prev
                    v_eff = (1.0 + r / 2.0) * v - (r / 2.0) * v_prev
                else:
                    v_eff = v
                x = x + v_eff * d
                v_prev = v
                d_prev = d
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "denoised": denoised})
        return x

    return sampler_function


class FlowExtrapSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0,
                              "step": 0.1}),
            "order": ("INT", {"default": 2, "min": 1, "max": 2}),
            "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                     "max": 10.0, "step": 0.1}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0,
                                  "max": 1.0, "step": 0.01}),
        },
        "optional": {
            "sigmas": ("SIGMAS",),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/flow"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, order, flow_shift, denoise, sigmas=None):
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"])
        # Safety net: if the channel fix left a Cosmos/Anima latent
        # at 4D, add the temporal dim the DiT expects.
        _expand5d = (_model_wants_temporal(model)
                     and latent_samples.ndim == 4)
        if _expand5d:
            latent_samples = latent_samples.unsqueeze(2)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if _expand5d:
            noise = noise.unsqueeze(2)
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
            src_label = f"grid(shift {flow_shift})"
        else:
            src_label = "external"
        sampler = comfy.samplers.KSAMPLER(
            _make_ab_sampler_function(order))
        print(f"[FlowExtrap] order={order}, sigmas={src_label}, "
              f"steps={steps}, cfg={cfg}")
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        out = dict(latent_image)
        if _expand5d:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out,)


def flow_sigma_grid(steps, shift, curve, curve_strength, denoise=1.0):
    """Sigma grid with a warpable parameterisation.

    curve 'linear' reproduces flow_linear_shift_sigmas. 'detail' bends
    the parameterisation so more steps land in the low-sigma detail
    region; 'smooth' densifies both ends. curve_strength blends the
    warp in (0 = linear).
    """
    t = torch.linspace(1.0, 0.0, steps + 1)
    if curve == "detail":
        warped = t ** 2.0
    elif curve == "smooth":
        warped = 0.5 - 0.5 * torch.cos(torch.pi * t)
    else:
        warped = t
    t = (1.0 - curve_strength) * t + curve_strength * warped
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    sigmas = sigmas * denoise
    sigmas[-1] = 0.0
    return sigmas


class FlowSigmaGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                     "max": 10.0, "step": 0.1}),
            "curve": (["linear", "detail", "smooth"],),
            "curve_strength": ("FLOAT", {"default": 0.5, "min": 0.0,
                                         "max": 1.0, "step": 0.05}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0,
                                  "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "make"
    CATEGORY = "sampling/flow"

    def make(self, steps, flow_shift, curve, curve_strength, denoise):
        sigmas = flow_sigma_grid(steps, flow_shift, curve,
                                 curve_strength, denoise)
        print(f"[FlowSigmaGrid] curve={curve}, strength="
              f"{curve_strength}, shift={flow_shift}, steps={steps}")
        return (sigmas,)



class FlowUnifiedSampler:
    """One node, dropdown algorithm: euler / momentum / ab2.

    strength is the momentum beta (ignored by euler and ab2).
    Optional sigmas input overrides the built-in grid, same as the
    standalone nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0,
                              "step": 0.1}),
            "algorithm": (["momentum", "ab2", "euler"],),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0,
                                   "max": 0.95, "step": 0.05}),
            "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                     "max": 10.0, "step": 0.1}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0,
                                  "max": 1.0, "step": 0.01}),
        },
        "optional": {
            "sigmas": ("SIGMAS",),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/flow"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, algorithm, strength, flow_shift, denoise,
               sigmas=None):
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"])
        # Safety net: if the channel fix left a Cosmos/Anima latent
        # at 4D, add the temporal dim the DiT expects.
        _expand5d = (_model_wants_temporal(model)
                     and latent_samples.ndim == 4)
        if _expand5d:
            latent_samples = latent_samples.unsqueeze(2)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if _expand5d:
            noise = noise.unsqueeze(2)
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
        if algorithm == "momentum":
            fn = _make_sampler_function(strength)
        elif algorithm == "ab2":
            fn = _make_ab_sampler_function(2)
        else:
            fn = _make_ab_sampler_function(1)
        sampler = comfy.samplers.KSAMPLER(fn)
        print(f"[FlowUnified] algorithm={algorithm}, "
              f"strength={strength}, steps={steps}, cfg={cfg}")
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        out = dict(latent_image)
        if _expand5d:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out,)






def _model_wants_temporal(model):
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    mod = type(dm).__module__ if dm is not None else ""
    return "cosmos" in mod or "anima" in mod


NODE_CLASS_MAPPINGS = {
    "FlowMomentumSampler": FlowMomentumSampler,
    "FlowExtrapSampler": FlowExtrapSampler,
    "FlowSigmaGrid": FlowSigmaGrid,
    "FlowUnifiedSampler": FlowUnifiedSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMomentumSampler": "Flow Momentum Sampler",
    "FlowExtrapSampler": "Flow Extrap Sampler (AB2)",
    "FlowSigmaGrid": "Flow Sigma Grid",
    "FlowUnifiedSampler": "Flow Sampler (Unified)",
}


"""Unit tests for the flow-momentum sampler math.

Run where torch exists (ComfyUI env or CI):
    python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes  # noqa: E402


def test_sigmas_descending_end_at_zero():
    s = nodes.flow_linear_shift_sigmas(30, 3.0, 1.0)
    assert len(s) == 31
    assert float(s[-1]) == 0.0
    assert bool((s[:-1] > s[1:]).all())
    assert abs(float(s[0]) - 1.0) < 1e-6


def test_sigmas_denoise_starts_lower():
    s = nodes.flow_linear_shift_sigmas(30, 3.0, 0.5)
    assert float(s[0]) < 0.9


def _const_velocity_model(c):
    def model(x, sigma, **kwargs):
        return x - sigma * c
    return model


def test_constant_velocity_field_is_exact_for_any_beta():
    # If the true velocity is constant, momentum must not distort it:
    # x_final = x0 + c * (sigma_last - sigma_first) = x0 - c * 1.0
    c = torch.full((1, 4), 0.3)
    x0 = torch.zeros(1, 4)
    sigmas = nodes.flow_linear_shift_sigmas(10, 3.0, 1.0)
    for beta in (0.0, 0.5, 0.9):
        fn = nodes._make_sampler_function(beta)
        out = fn(_const_velocity_model(c), x0.clone(), sigmas)
        assert torch.allclose(out, x0 - c, atol=1e-5), beta


def test_beta_zero_matches_plain_euler():
    # A curved field: denoised pulls toward a fixed target.
    target = torch.ones(1, 4)

    def model(x, sigma, **kwargs):
        return target

    x0 = torch.zeros(1, 4)
    sigmas = nodes.flow_linear_shift_sigmas(5, 3.0, 1.0)
    fn = nodes._make_sampler_function(0.0)
    out = fn(model, x0.clone(), sigmas)
    # manual euler on the same grid
    x = x0.clone()
    for i in range(len(sigmas) - 1):
        d = model(x, sigmas[i])
        v = (x - d) / sigmas[i]
        x = x + v * (sigmas[i + 1] - sigmas[i])
    assert torch.allclose(out, x, atol=1e-6)


def test_ab_order2_beats_euler_on_curved_field():
    # Velocity field v = sigma (independent of x): dx/dsigma = sigma,
    # exact integral from sigma=1 to 0 is -0.5. The multistep
    # extrapolation must land clearly closer than plain Euler on the
    # same grid (it is not exact: the terminal step jumps to the
    # model's x0 prediction, like every sampler here).
    def model(x, sigma, **kwargs):
        return x - sigma * sigma  # denoised -> v = (x - denoised)/sigma = sigma

    x0 = torch.zeros(1, 4)
    sigmas = nodes.flow_linear_shift_sigmas(10, 3.0, 1.0)
    out1 = nodes._make_ab_sampler_function(1)(model, x0.clone(), sigmas)
    out2 = nodes._make_ab_sampler_function(2)(model, x0.clone(), sigmas)
    err1 = abs(float(out1[0, 0]) - -0.5)
    err2 = abs(float(out2[0, 0]) - -0.5)
    assert err2 < err1
    assert err2 < 0.05


def test_grid_linear_matches_builtin():
    a = nodes.flow_sigma_grid(30, 3.0, "linear", 0.0)
    b = nodes.flow_linear_shift_sigmas(30, 3.0, 1.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_grid_detail_puts_more_steps_in_low_sigma():
    lin = nodes.flow_sigma_grid(30, 3.0, "linear", 0.0)
    det = nodes.flow_sigma_grid(30, 3.0, "detail", 1.0)
    assert float(det[-1]) == 0.0
    assert bool((det[:-1] > det[1:]).all())
    assert int((det < 0.5).sum()) > int((lin < 0.5).sum())

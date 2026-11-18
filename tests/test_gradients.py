"""Gradient checks for the minimal differentiable SRT prototype."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from diff_srt import brf_single_scattering, k_sphere, lai


def central_diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def test_brf_gradients_match_finite_difference():
    H, cover, d_L = 5.0, 0.5, 0.5
    mu_s = float(np.cos(np.deg2rad(45.0)))
    mu_v = float(np.cos(np.deg2rad(30.0)))

    def f(H_, cover_, dL_):
        return brf_single_scattering(
            H_, cover_, dL_, mu_s, mu_v, ro_soil=0.34, omega=0.9
        )

    grads_ad = jax.grad(f, argnums=(0, 1, 2))(H, cover, d_L)
    grads_fd = (
        central_diff(lambda x: f(x, cover, d_L), H),
        central_diff(lambda x: f(H, x, d_L), cover),
        central_diff(lambda x: f(H, cover, x), d_L),
    )

    for name, ad, fd in zip(("H", "cover", "d_L"), grads_ad, grads_fd):
        assert rel_err(float(ad), float(fd)) < 1e-5, (name, ad, fd)


def test_lai_gradient_matches_finite_difference():
    H, cover, d_L = 5.0, 0.5, 0.5

    def f(H_, cover_, dL_):
        return lai(H_, cover_, dL_)

    grads_ad = jax.grad(f, argnums=(0, 1, 2))(H, cover, d_L)
    grads_fd = (
        central_diff(lambda x: f(x, cover, d_L), H),
        central_diff(lambda x: f(H, x, d_L), cover),
        central_diff(lambda x: f(H, cover, x), d_L),
    )

    for name, ad, fd in zip(("H", "cover", "d_L"), grads_ad, grads_fd):
        assert rel_err(float(ad), float(fd)) < 1e-5, (name, ad, fd)


def test_k_sphere_properties():
    H, cover = 5.0, 0.5
    mu = float(np.cos(np.deg2rad(30.0)))

    # K(z, z) = 1
    for z in (1.0, 2.5, 4.0):
        assert abs(float(k_sphere(z, z, mu, H, cover)) - 1.0) < 1e-6

    # K in [0, 1] on a small grid
    for z1 in (1.0, 2.0, 3.0, 4.0):
        for z2 in (1.0, 2.0, 3.0, 4.0):
            k = float(k_sphere(z1, z2, mu, H, cover))
            assert -1e-6 <= k <= 1.0 + 1e-6, (z1, z2, k)


def test_lai_increases_with_cover_and_dL():
    base = float(lai(5.0, 0.5, 0.5))
    assert float(lai(5.0, 0.7, 0.5)) > base
    assert float(lai(5.0, 0.5, 0.7)) > base
    assert float(lai(5.0, 0.5, 0.5)) > 0.0

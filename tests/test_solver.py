"""Tests for the direct-beam Volterra solver."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from diff_srt import direct_beam


def central_diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def test_direct_beam_properties():
    z = jnp.linspace(0.0, 5.0, 21)
    mu_s = float(np.cos(np.deg2rad(45.0)))
    u, i = direct_beam(z, 5.0, 0.5, 1.0, mu_s)

    assert jnp.all(u > 0.0) and jnp.all(u <= 1.0 + 1e-9)
    assert jnp.all(i > 0.0) and jnp.all(i <= 1.0 + 1e-9)
    assert jnp.all(jnp.diff(u) <= 1e-9)
    assert jnp.all(jnp.diff(i) <= 1e-9)


def test_direct_beam_gradients():
    mu_s = float(np.cos(np.deg2rad(45.0)))

    def f(H, cover, d_L):
        # 20 nodes deliberately avoids the K-branch tangency that occurs
        # for this geometry at 21 nodes (z=2.5 is exactly mid-height).
        z = jnp.linspace(0.0, H, 20)
        u, i = direct_beam(z, H, cover, d_L, mu_s)
        return u[-1] + i[-1]

    H, cover, d_L = 5.0, 0.5, 1.0
    grads_ad = jax.grad(f, argnums=(0, 1, 2))(H, cover, d_L)
    grads_fd = (
        central_diff(lambda x: f(x, cover, d_L), H),
        central_diff(lambda x: f(H, x, d_L), cover),
        central_diff(lambda x: f(H, cover, x), d_L),
    )

    for name, ad, fd in zip(("H", "cover", "d_L"), grads_ad, grads_fd):
        rel = abs(float(ad) - float(fd)) / max(abs(float(fd)), 1e-12)
        assert rel < 1e-4, (name, ad, fd, rel)

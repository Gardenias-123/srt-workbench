"""Leaf-angle-distribution dependent optics (NumPy), ported from the Fortran.

* ``gfun`` - G-function, get_G_fun_v07.f / gfun_v07.f
* ``gind`` - area phase function, gind_v07.f + get_APF_v07.f

LAD codes (iorien): 1 planophile (a=1,b=2), 2 erectophile (-1,2),
3 plagiophile (-1,4), 4 extremophile (1,4), 5 uniform (0,2), 6 spherical,
7 horizontal, 8 vertical, 9 atmosphere.

For LAD 1-5 the phase function is linear in (ro, tau):
``gind = ro * A_refl + tau * A_trans``; the geometry-only matrices are
precomputed once with :func:`apf_matrices` (the leaf-normal integral is too
large to keep inside the JAX graph).
"""
from __future__ import annotations

import numpy as np

PI = 3.141593  # Fortran constant used by get_G / get_APF
_AB = {1: (1.0, 2.0), 2: (-1.0, 2.0), 3: (-1.0, 4.0), 4: (1.0, 4.0), 5: (0.0, 2.0)}


def _leaf_grid(n=30, m=None):
    m = m or 4 * n
    h_theta = 0.5 * PI / n
    h_fi = 2.0 * PI / m
    theta = (np.arange(n) + 0.5) * h_theta
    fi = (np.arange(m) + 0.5) * h_fi
    return theta, fi, h_theta, h_fi


def _theta_tran(theta, fi, alpha_slope):
    """Geotropic rotation of the leaf normal, folded to [0, pi/2]."""
    sinA, cosA = np.sin(alpha_slope), np.cos(alpha_slope)
    x_p = np.sin(theta)[:, None] * np.cos(fi)[None, :]
    z_p = np.cos(theta)[:, None]
    z = np.clip(-sinA * x_p + cosA * z_p, -1.0, 1.0)
    tt = np.arccos(z)
    return np.where(tt > PI / 2.0, PI - tt, tt)


def gfun(lad, mu, fi=None, alpha_slope=0.0):
    """G-function for all LAD types (vectorized over mu/fi)."""
    mu = np.asarray(mu, float)
    fi = np.zeros_like(mu) if fi is None else np.asarray(fi, float)
    if lad == 6:
        return np.full_like(mu, 0.5)
    if lad == 7:
        return np.abs(mu)
    if lad == 8:
        return 0.6366197 * np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    if lad == 9:
        return np.ones_like(mu)
    if lad not in _AB:
        raise NotImplementedError(f"LAD {lad}")
    a, b = _AB[lad]
    theta, fi_l, h_theta, h_fi = _leaf_grid()
    tt = _theta_tran(theta, fi_l, alpha_slope)
    w = (1.0 + a * np.cos(b * tt)) * h_fi
    st = np.sin(theta)
    smu = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    dot = np.abs(
        mu[..., None, None] * np.cos(theta)[None, :, None]
        + smu[..., None, None] * st[None, :, None]
        * np.cos(fi[..., None, None] - fi_l[None, None, :])
    )
    s = np.sum(dot * w[None, :, :], axis=(-1, -2))
    return s * h_theta / (PI * PI)


def apf_matrices(lad, alpha_slope, b1, f1, b2, f2):
    """Geometry-only phase matrices for LAD 1-5.

    Returns ``(A_refl, A_trans)`` with shape ``(len(b1), len(b2))`` such that
    ``gind = ro*A_refl + tau*A_trans`` reproduces get_APF.
    """
    a, b = _AB[lad]
    b1 = np.asarray(b1, float); f1 = np.asarray(f1, float)
    b2 = np.asarray(b2, float); f2 = np.asarray(f2, float)
    theta, fi_l, h_theta, h_fi = _leaf_grid()
    tt = _theta_tran(theta, fi_l, alpha_slope)
    w = (1.0 + a * np.cos(b * tt)) * (h_theta * h_fi / PI ** 3)
    st = np.sin(theta)

    def dots(bb, ff):
        smu = np.sqrt(np.maximum(1.0 - bb * bb, 0.0))
        return (bb[:, None, None] * np.cos(theta)[None, :, None]
                + smu[:, None, None] * st[None, :, None]
                * np.cos(ff[:, None, None] - fi_l[None, None, :]))

    d1 = dots(b1, f1)  # (Q,n,m)
    d2 = dots(b2, f2)  # (M,n,m)
    Q, M = len(b1), len(b2)
    A_refl = np.zeros((Q, M))
    A_trans = np.zeros((Q, M))
    for q in range(Q):
        prod = d1[q][None, :, :] * d2
        A_trans[q] = np.sum(np.maximum(prod, 0.0) * w, axis=(1, 2))
        A_refl[q] = np.sum(np.maximum(-prod, 0.0) * w, axis=(1, 2))
    return A_refl, A_trans


def gind_closed(lad, b1, f1, b2, f2, ro, tau):
    """Closed-form gind for LAD 6-9 (broadcasting)."""
    b1 = np.asarray(b1, float); f1 = np.asarray(f1, float)
    b2 = np.asarray(b2, float); f2 = np.asarray(f2, float)
    if lad == 6:
        s1 = np.sqrt(np.maximum(1.0 - b1 * b1, 0.0))
        s2 = np.sqrt(np.maximum(1.0 - b2 * b2, 0.0))
        cb = np.clip(b1 * b2 + s1 * s2 * np.cos(f2 - f1), -1.0, 1.0)
        ang = np.arccos(cb)
        sb = np.sqrt(np.maximum(1.0 - cb * cb, 0.0))
        wleaf = ro + tau
        return (wleaf * (sb - ang * cb) / (3.0 * PI) + tau * cb / 3.0) / PI
    if lad == 7:
        a = b1 * b2
        return np.where(a > 0, tau * a / PI, ro * np.abs(a) / PI)
    if lad == 8:
        beta = np.abs(f1 - f2)
        beta = np.where(beta > PI, beta - PI, beta)
        cb = np.cos(beta); sb = np.sin(beta)
        wleaf = ro + tau
        g = (wleaf * (sb - beta * cb) / (2.0 * PI) + tau * cb / 2.0)
        return g * np.sqrt(np.maximum(1.0 - b1 * b1, 0.0)) * np.sqrt(
            np.maximum(1.0 - b2 * b2, 0.0)) / PI
    if lad == 9:
        return 0.25 * (ro + tau) / PI
    raise NotImplementedError(f"LAD {lad}")


def gind(lad, b1, f1, b2, f2, ro, tau, alpha_slope=0.0):
    """General gind; LAD 1-5 computes the leaf-normal integral directly."""
    if lad in (6, 7, 8, 9):
        return gind_closed(lad, b1, f1, b2, f2, ro, tau)
    A_refl, A_trans = apf_matrices(lad, alpha_slope, b1, f1, b2, f2)
    return ro * A_refl + tau * A_trans

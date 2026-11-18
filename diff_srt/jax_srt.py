"""Differentiable JAX port of the full SRT/TopoSRT solver (flat, sphere).

This mirrors :mod:`diff_srt.ref_srt`, which is validated line by line against
the Fortran reference (``SRT_diag``).  The four scattering terms
``ScatDD/DU/UD/UU`` are used explicitly, which is the fix for the earlier
"single-branch" approximation that made the S problem 3x too large.

The quadrature (Carlson + trans_DS_to_Plane) is precomputed with NumPy for a
given SZA, because it does not depend on any canopy parameter.  Everything
that does (K, sigma, phase matrices, sources) is differentiable.

Conventions are identical to the Fortran: a direction is stored as an
"upward" pair ``(b>0, fi)``; the downward direction is ``(-b, fi+pi)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

EPSX = 1e-4


def _ratio(xx):
    """(1-exp(-xx))/xx with the Fortran epsx branch."""
    small = xx <= EPSX
    x = jnp.where(small, 1.0, xx)
    return jnp.where(small, 1.0, (1.0 - jnp.exp(-x)) / x)


def _eta(xx):
    return 1.0 - _ratio(xx)


def _safe_b(b):
    """Avoid division by zero for view directions at the local horizon."""
    tiny = jnp.abs(b) < 1e-8
    sign = jnp.where(b < 0, -1.0, 1.0)
    return jnp.where(tiny, sign * 1e-8, b)


# --------------------------------------------------------------------------
# Canopy structure
# --------------------------------------------------------------------------
def p_profile(z, H, cover, tree_type=4):
    u = jnp.clip(z / H, 0.0, 1.0)
    if tree_type == 1:
        return cover + 0.0 * u
    if tree_type == 2:
        return 1.0 - (1.0 - cover) ** (u ** 2)
    return 1.0 - (1.0 - cover) ** (4.0 * u * (1.0 - u))


def cond_pcf_sphere(z1, z2, mu, H, cover, alpha=0.0, fi=0.0):
    """cond_pcf case(4) with the Fortran's real(4) branch decisions.

    Arithmetic is float64 (so autodiff stays stable), but the quantities that
    enter the Fortran's discontinuous comparisons (lam, rz1, rz2) are rounded
    to float32 first: at crown tangency ``lam == rz2`` the real(4) side of the
    edge decides between two O(1)-different branches.
    """
    dt = jnp.result_type(z1, z2, mu, H, cover)
    f32 = jnp.float32
    one = 1.0
    epsz = 1e-3 * H
    epsmu = 0.0017
    mu_abs = jnp.abs(mu)
    one_minus = jnp.maximum(one - mu * mu, 0.0)
    sq = jnp.sqrt(one_minus)
    # Fortran: n = cos(fi)cos(alpha) + |mu| sin(alpha)/sqrt(1-mu^2).
    # Guard the mu -> 1 limit so no inf/nan enters the graph.
    ratio = jnp.where(sq > 0.0, mu_abs * jnp.sin(alpha) / jnp.where(sq > 0.0, sq, one), 0.0)
    n = jnp.cos(fi) * jnp.cos(alpha) + ratio
    lam = jnp.abs(z1 - z2) * sq / jnp.maximum(epsmu, mu_abs) * jnp.sqrt(n * n + jnp.sin(fi) ** 2)
    lam = jnp.where(mu_abs == one, 0.0, lam)
    lam = jnp.where(mu_abs == 0.0, jnp.abs(z1 - z2) / epsmu, lam)
    rz1 = jnp.sqrt(jnp.maximum(z1 * jnp.abs(H - z1), 1e-30))
    rz2 = jnp.sqrt(jnp.maximum(z2 * jnp.abs(H - z2), 1e-30))
    # real(4) rounding for the branch decisions only
    lam_c = lam.astype(f32).astype(dt)
    rz1 = rz1.astype(f32).astype(dt)
    rz2 = rz2.astype(f32).astype(dt)
    xrz = jnp.minimum(rz1, rz2)
    xrs = jnp.maximum(rz1, rz2)

    kappa_far = one - (xrz / jnp.maximum(rz1, 1e-30)) ** 2
    alfa = 2.0 * lam_c * xrs
    alfa = jnp.where(alfa > 0, (xrs * xrs - xrz * xrz + lam_c * lam_c) / jnp.maximum(alfa, 1e-30), one)
    alfa = jnp.arccos(jnp.clip(alfa, -1.0 + 1e-6, 1.0 - 1e-6))
    beta = 2.0 * lam_c * xrz
    beta = jnp.where(beta > 0, (xrz * xrz - xrs * xrs + lam_c * lam_c) / jnp.maximum(beta, 1e-30), one)
    beta = jnp.arccos(jnp.clip(beta, -1.0 + 1e-6, 1.0 - 1e-6))
    kappa_p = alfa * xrs * xrs + beta * xrz * xrz - lam_c * xrs * jnp.sin(alfa)
    kappa_p = kappa_p / jnp.maximum(jnp.pi * rz1 * rz1, 1e-30)
    kappa_p = one - kappa_p
    kappa = jnp.where(xrs >= xrz + lam_c, kappa_far,
                      jnp.where(lam_c >= xrs + xrz, one, kappa_p))

    u1 = z1 / H
    u2 = z2 / H
    pz1 = one - (one - cover) ** (4.0 * u1 * (one - u1))
    pz2 = one - (one - cover) ** (4.0 * u2 * (one - u2))
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / jnp.maximum(pz1, 1e-30)
    cpf = one - (one - pz2) * cpf
    z1b = (z1 <= epsz) | (jnp.abs(H - z1) <= epsz)
    z2b = (z2 <= epsz) | (jnp.abs(H - z2) <= epsz)
    cpf = jnp.where(z1b, jnp.where(lam_c <= rz2, one, pz2),
                    jnp.where(z2b, 0.0, cpf))
    return jnp.clip(cpf, 0.0, 1.0)


def _circle_kappa(dd, Rone=1.0, Rtwo=None):
    """Common circle-overlap kappa of cond_pcf cases 1-3."""
    one = 1.0
    alfa = 2.0 * dd * Rone
    alfa = (dd * dd + Rone * Rone - Rtwo * Rtwo) / jnp.maximum(alfa, 1e-30)
    alfa = jnp.arccos(jnp.clip(alfa, -1.0 + 1e-6, 1.0 - 1e-6))
    beta = 2.0 * dd * Rtwo
    beta = (dd * dd + Rtwo * Rtwo - Rone * Rone) / jnp.maximum(beta, 1e-30)
    beta = jnp.arccos(jnp.clip(beta, -1.0 + 1e-6, 1.0 - 1e-6))
    kp = (Rone + Rtwo - dd) * (dd + Rone - Rtwo) * (dd - Rone + Rtwo) * (dd + Rone + Rtwo)
    kp = 0.5 * jnp.sqrt(jnp.maximum(kp, 0.0))
    kp = Rone * Rone * alfa + Rtwo * Rtwo * beta - kp
    kp = 1.0 - kp / jnp.pi
    return jnp.where(dd >= Rone + Rtwo, one,
                     jnp.where(dd <= jnp.abs(Rone - Rtwo),
                               one - jnp.minimum(Rone * Rone, Rtwo * Rtwo), kp))


def _lam_slope(z1, z2, mu, alpha, fi, epsmu=0.0017):
    one = 1.0
    mu_abs = jnp.abs(mu)
    sq = jnp.sqrt(jnp.maximum(one - mu * mu, 0.0))
    ratio = jnp.where(sq > 0.0, mu_abs * jnp.sin(alpha) / jnp.where(sq > 0.0, sq, one), 0.0)
    n = jnp.cos(fi) * jnp.cos(alpha) + ratio
    lam = (jnp.abs(z1 - z2) * sq / jnp.maximum(epsmu, mu_abs)
           * jnp.sqrt(n * n + jnp.sin(fi) ** 2))
    lam = jnp.where(mu_abs == one, 0.0, lam)
    lam = jnp.where(mu_abs == 0.0, jnp.abs(z1 - z2) / epsmu, lam)
    return lam


def cond_pcf_cone(z1, z2, mu, H, cover, diam, alpha=0.0, fi=0.0, alpha_leaf=0.0):
    """cond_pcf case(2): conical crowns."""
    one = 1.0
    epsz = 1e-3 * H
    dt = jnp.result_type(z1, z2, mu, H, cover)
    lam = _lam_slope(z1, z2, mu, alpha, fi)
    cosA = jnp.cos(alpha_leaf)
    rh = (diam / 2.0) / H * cosA
    rz1 = rh * z1 / cosA
    rz2 = rh * z2 / cosA
    Rtwo = rz2 / jnp.maximum(rz1, 1e-30)
    dd = jnp.sqrt(jnp.maximum(one - jnp.cos(fi) ** 2 * jnp.sin(alpha_leaf) ** 2, 0.0))
    dd = dd * lam / jnp.maximum(rz1, 1e-30)
    f32 = jnp.float32
    lam_c = lam.astype(f32).astype(dt)
    dd_c = dd.astype(f32).astype(dt)
    kappa = _circle_kappa(dd_c, one, Rtwo)
    gap = one - cover
    u1 = z1 / H; u2 = z2 / H
    pz1 = one - gap ** (u1 * u1)
    pz2 = one - gap ** (u2 * u2)
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / jnp.maximum(pz1, 1e-30)
    cpf = one - (one - pz2) * cpf
    cpf = jnp.where(z1 <= epsz, jnp.where(lam_c <= rz2, one, pz2),
                    jnp.where(z2 <= epsz, 0.0, cpf))
    return jnp.clip(cpf, 0.0, 1.0)


def cond_pcf_ellipsoid(z1, z2, mu, H, cover, diam, alpha=0.0, fi=0.0,
                       alpha_leaf=0.0):
    """cond_pcf case(3): ellipsoidal crowns."""
    one = 1.0
    epsz = 1e-3 * H
    dt = jnp.result_type(z1, z2, mu, H, cover)
    lam = _lam_slope(z1, z2, mu, alpha, fi)
    cosA = jnp.cos(alpha_leaf)
    aratio = diam / H * cosA
    rz1 = z1 / cosA
    rz1 = rz1 * jnp.abs(H / cosA - z1 / cosA)
    rz1 = aratio * jnp.sqrt(jnp.maximum(rz1, 0.0))
    rz2 = z2 / cosA
    rz2 = rz2 * jnp.abs(H / cosA - z2 / cosA)
    rz2 = aratio * jnp.sqrt(jnp.maximum(rz2, 0.0))
    Rtwo = rz2 / jnp.maximum(rz1, 1e-30)
    dd = jnp.sqrt(jnp.maximum(one - jnp.cos(fi) ** 2 * jnp.sin(alpha_leaf) ** 2, 0.0))
    dd = dd * lam / jnp.maximum(rz1, 1e-30)
    f32 = jnp.float32
    lam_c = lam.astype(f32).astype(dt)
    dd_c = dd.astype(f32).astype(dt)
    kappa = _circle_kappa(dd_c, one, Rtwo)
    gap = one - cover
    u1 = z1 / H; u2 = z2 / H
    pz1 = one - gap ** (4.0 * u1 * (one - u1))
    pz2 = one - gap ** (4.0 * u2 * (one - u2))
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / jnp.maximum(pz1, 1e-30)
    cpf = one - (one - pz2) * cpf
    z1b = (z1 <= epsz) | (jnp.abs(H - z1) <= epsz)
    z2b = (z2 <= epsz) | (jnp.abs(H - z2) <= epsz)
    cpf = jnp.where(z1b, jnp.where(lam_c <= rz2, one, pz2),
                    jnp.where(z2b, 0.0, cpf))
    return jnp.clip(cpf, 0.0, 1.0)


def cond_pcf_cylinder(z1, z2, mu, H, cover, diam, alpha=0.0, fi=0.0,
                      alpha_leaf=0.0):
    """cond_pcf case(1) (cylinder).  float64 math, real(4) branch decisions."""
    dt = jnp.result_type(z1, z2, mu, H, cover)
    f32 = jnp.float32
    one = 1.0
    epsz = 1e-3 * H
    epsmu = 0.0017
    R = diam / 2.0
    mu_abs = jnp.abs(mu)
    sq = jnp.sqrt(jnp.maximum(one - mu * mu, 0.0))
    ratio = jnp.where(sq > 0.0, mu_abs * jnp.sin(alpha) / jnp.where(sq > 0.0, sq, one), 0.0)
    n = jnp.cos(fi) * jnp.cos(alpha) + ratio
    lam = jnp.abs(z1 - z2) * sq / jnp.maximum(epsmu, mu_abs) * jnp.sqrt(n * n + jnp.sin(fi) ** 2)
    lam = jnp.where(mu_abs == one, 0.0, lam)
    lam = jnp.where(mu_abs == 0.0, jnp.abs(z1 - z2) / epsmu, lam)
    dd = jnp.sqrt(jnp.maximum(one - jnp.cos(fi) ** 2 * jnp.sin(alpha_leaf) ** 2, 0.0)) * lam / R
    lam_c = lam.astype(f32).astype(dt)
    dd_c = dd.astype(f32).astype(dt)
    Rone = one
    Rtwo = one
    alfa = 2.0 * dd_c * Rone
    alfa = (dd_c * dd_c + Rone * Rone - Rtwo * Rtwo) / jnp.maximum(alfa, 1e-30)
    alfa = jnp.arccos(jnp.clip(alfa, -1.0 + 1e-6, 1.0 - 1e-6))
    beta = 2.0 * dd_c * Rtwo
    beta = (dd_c * dd_c + Rtwo * Rtwo - Rone * Rone) / jnp.maximum(beta, 1e-30)
    beta = jnp.arccos(jnp.clip(beta, -1.0 + 1e-6, 1.0 - 1e-6))
    kp = (Rone + Rtwo - dd_c) * (dd_c + Rone - Rtwo) * (dd_c - Rone + Rtwo) * (dd_c + Rone + Rtwo)
    kp = 0.5 * jnp.sqrt(jnp.maximum(kp, 0.0))
    kp = Rone * Rone * alfa + Rtwo * Rtwo * beta - kp
    kp = 1.0 - kp / jnp.pi
    kappa = jnp.where(dd_c >= Rone + Rtwo, one,
                      jnp.where(dd_c <= jnp.abs(Rone - Rtwo),
                                one - jnp.minimum(Rone * Rone, Rtwo * Rtwo), kp))
    pz1 = cover + jnp.zeros_like(z1)
    pz2 = cover + jnp.zeros_like(z2)
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / jnp.maximum(pz1, 1e-30)
    cpf = one - (one - pz2) * cpf
    cpf = jnp.where(z1 <= epsz, jnp.where(lam_c <= R, one, pz2),
                    jnp.where(z2 <= epsz, 0.0, cpf))
    return jnp.clip(cpf, 0.0, 1.0)


def cond_pcf(z1, z2, mu, H, cover, diam=5.0, tree_type=4, alpha=0.0, fi=0.0,
             alpha_leaf=None):
    if alpha_leaf is None:
        alpha_leaf = alpha
    if tree_type == 4:
        return cond_pcf_sphere(z1, z2, mu, H, cover, alpha, fi)
    if tree_type == 1:
        return cond_pcf_cylinder(z1, z2, mu, H, cover, diam, alpha, fi,
                                 alpha_leaf=alpha_leaf)
    if tree_type == 2:
        return cond_pcf_cone(z1, z2, mu, H, cover, diam, alpha, fi, alpha_leaf)
    if tree_type == 3:
        return cond_pcf_ellipsoid(z1, z2, mu, H, cover, diam, alpha, fi, alpha_leaf)
    raise NotImplementedError(f"cond_pcf tree_type={tree_type}")


def kernel_arrays(zcel, b, H, cover, fi=None, alpha=0.0, diam=5.0, tree_type=4,
                  alpha_leaf=None):
    """K (nfi,nz,nz) and kkH (nfi,nz) exactly as get_kernel.f."""
    nz = zcel.shape[0] - 1
    zi = zcel[:-1]
    zip1 = zcel[1:]
    if fi is None:
        fi = jnp.zeros_like(b)
    fi = jnp.asarray(fi)
    b3 = b[:, None, None]
    f3 = fi[:, None, None]
    z1a = zi[None, :, None]
    z2a = zi[None, None, :]
    ki = cond_pcf(z1a, z2a, b3, H, cover, diam, tree_type, alpha, f3, alpha_leaf)
    ki1 = cond_pcf(zip1[None, :, None], z2a, b3, H, cover, diam, tree_type, alpha, f3, alpha_leaf)
    ki2 = cond_pcf(zip1[None, :, None], zip1[None, None, :], b3, H, cover, diam, tree_type, alpha, f3, alpha_leaf)
    ki3 = cond_pcf(z1a, zip1[None, None, :], b3, H, cover, diam, tree_type, alpha, f3, alpha_leaf)
    K = (ki + ki1 + ki2 + ki3) / 4.0
    diag = (ki + ki1 + ki2) / 3.0
    idx = jnp.arange(nz)
    K = K.at[:, idx, idx].set(diag[:, idx, idx])
    KH = (ki[:, idx, idx] + ki2[:, idx, idx] + ki3[:, idx, idx]) / 3.0
    return K, KH


# --------------------------------------------------------------------------
# Phase function / scattering matrices
# --------------------------------------------------------------------------
def gind_closed(lad, b1, f1, b2, f2, ro, tau):
    """Closed-form area phase function for LAD 6-9 (Fortran pi=3.141593)."""
    pi = 3.141593
    if lad == 6:
        s1 = jnp.sqrt(jnp.maximum(1.0 - b1 * b1, 1e-30))
        s2 = jnp.sqrt(jnp.maximum(1.0 - b2 * b2, 1e-30))
        cb = jnp.clip(b1 * b2 + s1 * s2 * jnp.cos(f2 - f1), -1.0, 1.0)
        ang = jnp.arccos(cb)
        sb = jnp.sqrt(jnp.maximum(1.0 - cb * cb, 1e-30))
        wleaf = ro + tau
        return (wleaf * (sb - ang * cb) / (3.0 * pi) + tau * cb / 3.0) / pi
    if lad == 7:
        a = b1 * b2
        return jnp.where(a > 0, tau * a / pi, ro * jnp.abs(a) / pi)
    if lad == 8:
        beta = jnp.abs(f1 - f2)
        beta = jnp.where(beta > pi, beta - pi, beta)
        cb = jnp.cos(beta)
        sb = jnp.sin(beta)
        wleaf = ro + tau
        g = (wleaf * (sb - beta * cb) / (2.0 * pi) + tau * cb / 2.0)
        return (g * jnp.sqrt(jnp.maximum(1.0 - b1 * b1, 1e-30))
                * jnp.sqrt(jnp.maximum(1.0 - b2 * b2, 1e-30)) / pi)
    if lad == 9:
        return 0.25 * (ro + tau) / pi
    raise NotImplementedError(f"LAD {lad}")


def gind_sphere(b1, f1, b2, f2, ro, tau):
    s1 = jnp.sqrt(jnp.maximum(1.0 - b1 * b1, 1e-30))
    s2 = jnp.sqrt(jnp.maximum(1.0 - b2 * b2, 1e-30))
    cb = jnp.clip(b1 * b2 + s1 * s2 * jnp.cos(f2 - f1), -1.0, 1.0)
    ang = jnp.arccos(cb)
    sb = jnp.sqrt(jnp.maximum(1.0 - cb * cb, 1e-30))
    wleaf = ro + tau
    return (wleaf * (sb - ang * cb) / (3.0 * jnp.pi) + tau * cb / 3.0) / jnp.pi


def scatter_matrices(bc, fic, bu, fiu, ro, tau, dL, lad=6, apf=None):
    """(nfi_c, nfi_out) matrices for DD, DU, UD, UU (see get_Scat_coeff)."""
    BC = bc[:, None]
    FC = fic[:, None]
    BD = bu[None, :]
    FD = fiu[None, :]
    if lad in (6, 7, 8, 9):
        DD = dL * gind_closed(lad, -BC, FC + jnp.pi, -BD, FD + jnp.pi, ro, tau)
        DU = dL * gind_closed(lad, -BC, FC + jnp.pi, BD, FD, ro, tau)
        UD = dL * gind_closed(lad, BC, FC, -BD, FD + jnp.pi, ro, tau)
        UU = dL * gind_closed(lad, BC, FC, BD, FD, ro, tau)
    else:
        DD = dL * (ro * apf["DD_refl"] + tau * apf["DD_trans"])
        DU = dL * (ro * apf["DU_refl"] + tau * apf["DU_trans"])
        UD = dL * (ro * apf["UD_refl"] + tau * apf["UD_trans"])
        UU = dL * (ro * apf["UU_refl"] + tau * apf["UU_trans"])
    return DD, DU, UD, UU


def scatter_sun(b, fi, mu0, fi0, ro, tau, dL, lad=6, apf0=None):
    if lad in (6, 7, 8, 9):
        SCar0D = dL * gind_closed(lad, -mu0, fi0, -b, fi + jnp.pi, ro, tau)
        SCar0U = dL * gind_closed(lad, -mu0, fi0, b, fi, ro, tau)
    else:
        SCar0D = dL * (ro * apf0["D_refl"][0] + tau * apf0["D_trans"][0])
        SCar0U = dL * (ro * apf0["U_refl"][0] + tau * apf0["U_trans"][0])
    return SCar0D, SCar0U


def hotspot_factor(hotspot, mu0d, fi0d, mu, fi, sig0, sig):
    """Port of get_HSF (Kuusk), vectorized."""
    eps = 0.01745
    mu = jnp.asarray(mu)
    fi = jnp.asarray(fi)
    active = (hotspot > 0.0) & (mu > eps)
    safe_mu = jnp.where(active, mu, 1.0)
    cosgamma = mu0d * safe_mu + jnp.sqrt(jnp.maximum((1.0 - mu0d ** 2) * (1.0 - safe_mu ** 2), 1e-30)) * jnp.cos(fi - fi0d)
    delta = mu0d ** 2 + safe_mu ** 2 + 2.0 * cosgamma * jnp.abs(safe_mu * mu0d)
    delta = jnp.sqrt(jnp.abs(delta)) / jnp.abs(safe_mu * mu0d)
    x = hotspot * delta
    x = jnp.where(x <= EPSX, 1.0, (1.0 - jnp.exp(-jnp.where(x <= EPSX, 1.0, x)))
                  / jnp.where(x <= EPSX, 1.0, x))
    fac = jnp.sqrt(jnp.abs(sig0 * safe_mu) / jnp.abs(sig * mu0d))
    hsf = jnp.abs(1.0 - fac * x)
    return jnp.where(active, hsf, 1.0)


# --------------------------------------------------------------------------
# Solver kernels
# --------------------------------------------------------------------------
def kernel_tables(K, kkH):
    """Precompute the K tables used by every Volterra call of a case."""
    Kd = jnp.einsum("mii->mi", K)
    kD = K / Kd[:, None, :]
    kU = K[:, ::-1, ::-1] / kkH[:, ::-1][:, None, :]
    return Kd, kD, kU


def volterra(zcel, lamD, lamU, f, fH, K, kkH, tables=None):
    """Volterra_lin as a single ``lax.scan`` over the nz layers.

    The i-loop of the Fortran recurrence is sequential; compiling it unrolled
    (20 copies x ~8 calls per case) dominates XLA compile time and reverse-mode
    memory.  A scan keeps the graph O(1) in nz while preserving the exact
    algebra (the inner j-sum is a masked reduction, as before).
    """
    nz = zcel.shape[0] - 1
    h = jnp.diff(zcel)
    nfi = lamD.shape[0]
    if tables is None:
        Kd, kD, kU = kernel_tables(K, kkH)
    else:
        Kd, kD, kU = tables
    j0 = jnp.arange(nz)
    jh0 = nz - 1 - j0
    # per-layer constants for the scan
    zargD = (lamD[:, None] * Kd) * h[None, :]
    zargU = (lamU[:, None] * kkH[:, jh0]) * h[jh0][None, :]
    u0 = jnp.zeros((nz + 1, nfi)).at[0].set(f[0])
    uH0 = jnp.zeros((nz + 1, nfi)).at[nz].set(fH[nz])
    psi0 = jnp.zeros((nz, nfi))
    psiH0 = jnp.zeros((nz, nfi))
    idx = jnp.arange(nz)

    xs = (
        jnp.arange(1, nz + 1),                 # i
        f[1:],                                  # f[i]
        fH[:nz][::-1],                          # fH[nz-i]
        h,                                      # h[i-1]
        Kd.T,                                   # Kd[:, i-1]
        jnp.transpose(kD, (1, 0, 2)),           # kD[:, i-1, :]
        jnp.transpose(kU, (1, 0, 2)),           # kU[:, i-1, :]
        kkH[:, ::-1].T,                         # kkH[:, nz-i]
        h[::-1],                                # h[nz-i]
    )

    def body(carry, xs):
        u, uH, psi, psiH = carry
        i, f_i, fH_i1, h_prev, Kd_i, kD_row, kU_row, kkH_i1, h_i1 = xs
        i1 = nz - i
        psi = psi.at[i - 1].set(f_i)
        psiH = psiH.at[i1].set(fH_i1)
        mask_row = j0 < (i - 1)
        A = (u[:-1].T * (1.0 - jnp.exp(-zargD))
             + (psi.T - u[:-1].T) * _eta(zargD))
        contrib = jnp.sum(jnp.where(mask_row[None, :], kD_row * A, 0.0), axis=1)
        psi = psi.at[i - 1].add(-contrib)
        BU = (uH[jh0 + 1].T * (1.0 - jnp.exp(-zargU))
              + (psiH[jh0].T - uH[jh0 + 1].T) * _eta(zargU))
        contribU = jnp.sum(jnp.where(mask_row[None, :], kU_row * BU, 0.0), axis=1)
        psiH = psiH.at[i1].add(-contribU)

        xx = lamD * Kd_i * h_prev
        unew = u[i - 1] * jnp.exp(-xx) + (psi[i - 1] - u[i - 1]) * _ratio(xx)
        neg = unew < 0.0
        psi = psi.at[i - 1].set(jnp.where(neg, u[i - 1], psi[i - 1]))
        u = u.at[i].set(jnp.where(neg, u[i - 1] * jnp.exp(-xx), unew))
        xx = lamU * kkH_i1 * h_i1
        uHnew = uH[i1 + 1] * jnp.exp(-xx) + (psiH[i1] - uH[i1 + 1]) * _ratio(xx)
        neg = uHnew < 0.0
        psiH = psiH.at[i1].set(jnp.where(neg, uH[i1 + 1], psiH[i1]))
        uH = uH.at[i1].set(jnp.where(neg, uH[i1 + 1] * jnp.exp(-xx), uHnew))
        return (u, uH, psi, psiH), None

    (u, uH, psi, psiH), _ = jax.lax.scan(body, (u0, uH0, psi0, psiH0), xs)
    return u, uH, psi, psiH


def get_idir(zcel, p, lam0, Kd0, psi0, udir):
    nz = zcel.shape[0] - 1
    h = jnp.diff(zcel)
    idir = jnp.zeros(nz + 1).at[0].set(1.0)
    xxpsi = 0.0
    for i in range(1, nz + 1):
        pmean = 0.5 * (p[i - 1] + p[i])
        zarg = lam0 * Kd0[i - 1] * h[i - 1]
        eta = udir[i - 1] * (1.0 - jnp.exp(-zarg)) + (psi0[i - 1] - udir[i - 1]) * _eta(zarg)
        xxpsi = xxpsi + pmean * eta / Kd0[i - 1]
        idir = idir.at[i].set(1.0 - xxpsi)
    return idir


def get_f0_fi0(zcel, p, K, kkH, lam0, SCar0D, SCar0U, udir, idir,
               psi0, d0, dH, i0, b):
    """get_f0_Fi0_v02.f, vectorized over layers."""
    nz = zcel.shape[0] - 1
    nfi = b.shape[0]
    h = jnp.diff(zcel)
    lam0bl = lam0 * _safe_b(b)
    xxD = i0 * SCar0D / lam0bl
    xxU = i0 * SCar0U / lam0bl
    Kd = jnp.einsum("mii->mi", K)
    idx = jnp.arange(nz)
    j0 = jnp.arange(nz)
    jh0 = nz - 1 - j0
    mask_le = j0[:, None] <= j0[None, :]
    zargD = (lam0 * Kd) * h[None, :]
    etaD = (udir[None, :-1] * (1.0 - jnp.exp(-zargD))
            + (psi0[None, :] - udir[None, :-1]) * _eta(zargD))
    zargU = (lam0 * Kd[:, jh0]) * h[jh0][None, :]
    etaU = (udir[None, jh0] * (1.0 - jnp.exp(-zargU))
            + (psi0[None, jh0] - udir[None, jh0]) * _eta(zargU))
    kD = K / Kd[:, None, :]
    fD0i = jnp.sum(jnp.where(mask_le.T[None], kD * etaD[:, None, :], 0.0), axis=2).T
    Kr = K[:, ::-1, ::-1]
    kU = Kr / Kd[:, ::-1][:, None, :]
    kU = kU.at[:, idx, idx].set(kkH[:, ::-1] / Kd[:, ::-1])
    fU0i = jnp.sum(jnp.where(mask_le.T[None], kU * etaU[:, None, :], 0.0), axis=2).T
    fiD0 = jnp.zeros((nz + 1, nfi)).at[0].set(d0)
    fiU0 = jnp.zeros((nz + 1, nfi)).at[nz].set(dH)
    fD0 = jnp.zeros((nz + 1, nfi)).at[0].set(d0)
    fU0 = jnp.zeros((nz + 1, nfi)).at[nz].set(dH)
    for i in range(1, nz + 1):
        fiD0 = fiD0.at[i].set(xxD * (1.0 - idir[i]) + d0)
        fiU0 = fiU0.at[i - 1].set(xxU * (idir[i - 1] - idir[nz]) + dH)
        fD0 = fD0.at[i].set(fD0i[i - 1] * xxD + d0)
        fU0 = fU0.at[nz - i].set(fU0i[i - 1] * xxU + dH)
    return fD0, fU0, fiD0, fiU0


def layer_etas(zcel, u, uH, psi, psiH, lamD, lamU, K, kkH):
    nz = zcel.shape[0] - 1
    h = jnp.diff(zcel)
    Kd = jnp.einsum("mii->mi", K)
    zargD = (lamD * Kd.T) * h[:, None]
    etaD = jnp.where(
        zargD <= EPSX,
        0.5 * (u[:-1] + psi) * h[:, None],
        (u[:-1] * (1.0 - jnp.exp(-zargD))
         + (psi - u[:-1]) * _eta(zargD)) / jnp.maximum(lamD * Kd.T, 1e-30),
    )
    zargU = (lamU * kkH.T) * h[:, None]
    etaU = jnp.where(
        zargU <= EPSX,
        0.5 * (uH[1:] + psiH) * h[:, None],
        (uH[1:] * (1.0 - jnp.exp(-zargU))
         + (psiH - uH[1:]) * _eta(zargU)) / jnp.maximum(lamU * kkH.T, 1e-30),
    )
    return etaD, etaU


def _xx_from_etas(etaD, etaU, w0, SDD, SDU, SUU, SUD):
    eD = etaD * w0[None, :]
    eU = etaU * w0[None, :]
    return eU @ SUD + eD @ SDD, eU @ SUU + eD @ SDU


def get_fk(zcel, b_out, K_out, kkH_out, Scat, w0, lam1D, lam1U, K1, kkH1,
           u1, uH1, psi1, psiH1):
    nz = zcel.shape[0] - 1
    nfi_out = b_out.shape[0]
    SDD, SDU, SUD, SUU = Scat  # scatter_matrices order: DD, DU, UD, UU
    etaD, etaU = layer_etas(zcel, u1, uH1, psi1, psiH1, lam1D, lam1U, K1, kkH1)
    xxD, xxU = _xx_from_etas(etaD, etaU, w0, SDD, SDU, SUU, SUD)
    tril = jnp.tril(jnp.ones((nz, nz)))
    b_safe = _safe_b(b_out)
    fDk = jnp.einsum("mij,jm->im", K_out * tril[None], xxD) / b_safe[None, :]
    idx = jnp.arange(nz)
    Kup = K_out.at[:, idx, idx].set(kkH_out)
    triu = jnp.triu(jnp.ones((nz, nz)))
    fUk = jnp.einsum("mrc,cm->rm", Kup * triu[None], xxU) / b_safe[None, :]
    fDk = jnp.vstack([jnp.zeros((1, nfi_out)), fDk])
    fUk = jnp.vstack([fUk, jnp.zeros((1, nfi_out))])
    return fDk, fUk


def get_intpS(zcel, p, b_out, Scat, w0, lam1D, lam1U, K1, kkH1,
              u1, uH1, psi1, psiH1):
    nz = zcel.shape[0] - 1
    nfi_out = b_out.shape[0]
    SDD, SDU, SUD, SUU = Scat  # scatter_matrices order: DD, DU, UD, UU
    etaD, etaU = layer_etas(zcel, u1, uH1, psi1, psiH1, lam1D, lam1U, K1, kkH1)
    xxD, xxU = _xx_from_etas(etaD, etaU, w0, SDD, SDU, SUU, SUD)
    pmean = 0.5 * (p[:-1] + p[1:])
    b_safe = _safe_b(b_out)
    ssD = jnp.vstack([jnp.zeros((1, nfi_out)),
                      jnp.cumsum(pmean[:, None] * xxD, axis=0)]) / b_safe[None, :]
    termU = pmean[:, None] * xxU
    rev = jnp.flipud(jnp.cumsum(jnp.flipud(termU), axis=0)) / b_safe[None, :]
    ssU = jnp.vstack([rev, jnp.zeros((1, nfi_out))])
    return ssD, ssU


def get_idif(zcel, p, lamD, lamU, fiD0, fiU0, ssD, ssU, u, uH, psi, psiH,
             K, kkH):
    nz = zcel.shape[0] - 1
    h = jnp.diff(zcel)
    nfi = u.shape[1]
    idifD = jnp.zeros((nz + 1, nfi))
    idifU = jnp.zeros((nz + 1, nfi))
    idifD = idifD.at[0].set(fiD0[0])
    idifU = idifU.at[nz].set(fiU0[nz])
    Kd = jnp.einsum("mii->mi", K)
    xxpsi = jnp.zeros(nfi)
    xxpsih = jnp.zeros(nfi)
    for i in range(1, nz + 1):
        pmean = 0.5 * (p[i - 1] + p[i])
        zarg = lamD * Kd[:, i - 1] * h[i - 1]
        eta = u[i - 1] * (1.0 - jnp.exp(-zarg)) + (psi[i - 1] - u[i - 1]) * _eta(zarg)
        eta = eta / Kd[:, i - 1]
        xxpsi = xxpsi + pmean * eta
        idifD = idifD.at[i].set(fiD0[i] + ssD[i] - xxpsi)
        ih = nz - i
        pmean = 0.5 * (p[ih] + p[ih + 1])
        zarg = lamU * kkH[:, ih] * h[ih]
        eta = uH[ih + 1] * (1.0 - jnp.exp(-zarg)) + (psiH[ih] - uH[ih + 1]) * _eta(zarg)
        eta = eta / kkH[:, ih]
        xxpsih = xxpsih + pmean * eta
        idifU = idifU.at[ih].set(fiU0[ih] + ssU[ih] - xxpsih)
    return idifD, idifU


def get_h_fluxes(b, w0, IdifD, IdifU):
    fluxD = jnp.sum(b[None, :] * w0[None, :] * IdifD, axis=1)
    fluxU = jnp.sum(b[None, :] * w0[None, :] * IdifU, axis=1)
    return fluxD, fluxU


def norm_volterra(zcel, p, w0, lamD, lamU, sigD, sigU, K, kkH,
                  u, uH, psi, psiH):
    nz = zcel.shape[0] - 1
    h = jnp.diff(zcel)
    Kd = jnp.einsum("mii->mi", K)
    norm = 0.0
    for i in range(1, nz + 1):
        zargD = lamD * Kd[:, i - 1] * h[i - 1]
        xxD = u[i - 1] * (1.0 - jnp.exp(-zargD)) + (psi[i - 1] - u[i - 1]) * _eta(zargD)
        xxD = jnp.where(sigD < lamD, sigD * xxD / jnp.maximum(lamD * Kd[:, i - 1], 1e-30), 0.0)
        zargU = lamU * kkH[:, i - 1] * h[i - 1]
        xxU = uH[i] * (1.0 - jnp.exp(-zargU)) + (psiH[i - 1] - uH[i]) * _eta(zargU)
        xxU = jnp.where(sigU < lamU, sigU * xxU / jnp.maximum(lamU * kkH[:, i - 1], 1e-30), 0.0)
        norm = norm + 0.5 * (p[i - 1] + p[i]) * jnp.sum((xxU + xxD) * w0)
    return norm


# --------------------------------------------------------------------------
# Case assembly / driver
# --------------------------------------------------------------------------
def build_case(H, cover, dL, ro, tau, hotspot, mu0, fi0, quad, b_usr, fi_usr,
               nz=20, alpha=0.0, tree_type=4, diam=5.0, lad=6,
               alpha_leaf=None, apf=None, apf_usr=None, apf0=None,
               apf0_usr=None, gfac=None, fi0_hot=None):
    """Precompute the direction-dependent objects shared by all sub-problems.

    ``quad`` = (b, fi, w0) fixed Carlson quadrature (NumPy arrays).  ``fi0`` is
    the solar azimuth used by the phase function (Fortran real(4) roundtrip);
    ``fi0_hot`` is the value used by the hot-spot factor.
    """
    b, fi, w0 = (jnp.asarray(q) for q in quad)
    if alpha_leaf is None:
        alpha_leaf = alpha
    zcel = jnp.linspace(0.0, H, nz + 1)
    p = p_profile(zcel, H, cover, tree_type)
    K, kkH = kernel_arrays(zcel, b, H, cover, fi=fi, alpha=alpha,
                           diam=diam, tree_type=tree_type, alpha_leaf=alpha_leaf)
    Kusr, kkHusr = kernel_arrays(zcel, b_usr, H, cover, fi=fi_usr, alpha=alpha,
                                 diam=diam, tree_type=tree_type,
                                 alpha_leaf=alpha_leaf)
    K0, kkH0 = kernel_arrays(zcel, jnp.asarray([mu0]), H, cover,
                             fi=jnp.asarray([fi0 - jnp.pi]), alpha=alpha,
                             diam=diam, tree_type=tree_type,
                             alpha_leaf=alpha_leaf)
    K0, kkH0 = K0[0], kkH0[0]
    SCar = scatter_matrices(b, fi, b, fi, ro, tau, dL, lad, apf)
    SUsr = scatter_matrices(b, fi, b_usr, fi_usr, ro, tau, dL, lad, apf_usr)
    SCar0D, SCar0U = scatter_sun(b, fi, mu0, fi0, ro, tau, dL, lad, apf0)
    SUsr0D, SUsr0U = scatter_sun(b_usr, fi_usr, mu0, fi0, ro, tau, dL, lad,
                                 apf0_usr)
    # G-function values (host-precomputed when LAD != 6)
    if gfac is None:
        g_down = jnp.full_like(b, 0.5)
        g_up = jnp.full_like(b, 0.5)
        g_udown = jnp.full_like(b_usr, 0.5)
        g_uup = jnp.full_like(b_usr, 0.5)
        g_sun = 0.5
    else:
        g_down = jnp.asarray(gfac["dir_down"])
        g_up = jnp.asarray(gfac["dir_up"])
        g_udown = jnp.asarray(gfac["usr_down"])
        g_uup = jnp.asarray(gfac["usr_up"])
        g_sun = jnp.asarray(gfac["sun"])
    sig0 = dL * g_sun
    lam0 = sig0 / mu0
    one = jnp.ones((nz + 1, 1))
    tab0 = kernel_tables(K0[None], kkH0[None])
    udir, _, psi0, _ = volterra(zcel, jnp.array([lam0]), jnp.array([lam0]),
                                one, one, K0[None], kkH0[None], tables=tab0)
    udir, psi0 = udir[:, 0], psi0[:, 0]
    idir = get_idir(zcel, p, lam0, jnp.diag(K0), psi0, udir)
    tab = kernel_tables(K, kkH)
    tab_usr = kernel_tables(Kusr, kkHusr)
    return dict(zcel=zcel, p=p, b=b, fi=fi, w0=w0, b_usr=b_usr, fi_usr=fi_usr,
                tab=tab, tab_usr=tab_usr,
                K=K, kkH=kkH, Kusr=Kusr, kkHusr=kkHusr, K0=K0, kkH0=kkH0,
                SCar=SCar, SUsr=SUsr, SCar0D=SCar0D, SCar0U=SCar0U,
                SUsr0D=SUsr0D, SUsr0U=SUsr0U, sig0=sig0, lam0=lam0,
                udir=udir, psi0=psi0, idir=idir, mu0=mu0,
                fi0=fi0_hot if fi0_hot is not None else fi0,
                dL=dL, ro=ro, tau=tau, hotspot=hotspot, nz=nz, alpha=alpha,
                tree_type=tree_type, diam=diam, lad=lad,
                g_down=g_down, g_up=g_up, g_udown=g_udown, g_uup=g_uup,
                g_sun=g_sun)


def solve_problem(case, problem="BS_dir", maxiter=10):
    """Run one SRT sub-problem on a prebuilt case."""
    zcel = case["zcel"]; p = case["p"]
    b = case["b"]; w0 = case["w0"]; b_usr = case["b_usr"]
    K = case["K"]; kkH = case["kkH"]; Kusr = case["Kusr"]; kkHusr = case["kkHusr"]
    SCar = case["SCar"]; SUsr = case["SUsr"]
    SCar0D = case["SCar0D"]; SCar0U = case["SCar0U"]
    SUsr0D = case["SUsr0D"]; SUsr0U = case["SUsr0U"]
    lam0 = case["lam0"]; udir = case["udir"]; psi0 = case["psi0"]; idir = case["idir"]
    mu0 = case["mu0"]; fi0 = case["fi0"]
    dL = case["dL"]; hotspot = case["hotspot"]; nz = case["nz"]
    g_sun = case["g_sun"]
    g_down = case["g_down"]; g_up = case["g_up"]
    g_udown = case["g_udown"]; g_uup = case["g_uup"]
    sig0 = dL * g_sun
    zeros_b = jnp.zeros_like(b)
    zeros_u = jnp.zeros_like(b_usr)
    if problem == "BS_dir":
        ht = hotspot_factor(hotspot, -mu0, fi0, b, case["fi"], sig0, dL * g_up)
        sigCarD = dL * g_down
        sigCarU = dL * g_up * ht
        htu = hotspot_factor(hotspot, -mu0, fi0, b_usr, case["fi_usr"],
                             sig0, dL * g_uup)
        sigUsrD = dL * g_udown
        sigUsrU = dL * g_uup * htu
        i0 = 1.0
        d0 = zeros_b; dH = zeros_b
        d0u = zeros_u; dHu = zeros_u
    elif problem == "BS_dif":
        sigCarD = dL * g_down
        sigCarU = dL * g_up
        sigUsrD = dL * g_udown
        sigUsrU = dL * g_uup
        i0 = 0.0
        d0 = jnp.full_like(b, 1.0 / jnp.pi); dH = zeros_b
        d0u = jnp.full_like(b_usr, 1.0 / jnp.pi); dHu = zeros_u
    elif problem == "S":
        sigCarD = dL * g_down
        sigCarU = dL * g_up
        sigUsrD = dL * g_udown
        sigUsrU = dL * g_uup
        i0 = 0.0
        d0 = zeros_b; dH = jnp.full_like(b, 1.0 / jnp.pi)
        d0u = zeros_u; dHu = jnp.full_like(b_usr, 1.0 / jnp.pi)
    else:
        raise ValueError(problem)

    lamCarD = sigCarD / b
    lamCarU = sigCarU / b
    lamUsrD = sigUsrD / b_usr
    lamUsrU = sigUsrU / b_usr

    fD0, fU0, fiD0, fiU0 = get_f0_fi0(zcel, p, K, kkH, lam0, SCar0D, SCar0U,
                                      udir, idir, psi0, d0, dH, i0, b)
    fUsrD0, fUsrU0, fiUsrD0, fiUsrU0 = get_f0_fi0(
        zcel, p, Kusr, kkHusr, lam0, SUsr0D, SUsr0U, udir, idir, psi0,
        d0u, dHu, i0, b_usr)

    tab = case["tab"]; tab_usr = case["tab_usr"]
    vDk, vUk, psi, psiH = volterra(zcel, lamCarD, lamCarU, fD0, fU0, K, kkH,
                                    tables=tab)
    STCarD = jnp.zeros_like(vDk)
    STCarU = jnp.zeros_like(vUk)
    epsv0 = norm_volterra(zcel, p, w0, lamCarD, lamCarU, sigCarD, sigCarU,
                          K, kkH, vDk, vUk, psi, psiH)
    norm = epsv0
    reldel = jnp.array(1.0)
    # fixed 5 iterations: matches maxiter=10 starting at iter=6 in the Fortran
    for it in range(5):
        fDk, fUk = get_fk(zcel, b, K, kkH, SCar, w0, lamCarD, lamCarU,
                          K, kkH, vDk, vUk, psi, psiH)
        vDk, vUk, psi, psiH = volterra(zcel, lamCarD, lamCarU, fDk, fUk, K, kkH,
                                        tables=tab)
        STCarD = STCarD + fDk
        STCarU = STCarU + fUk
        epsv1 = norm_volterra(zcel, p, w0, lamCarD, lamCarU, sigCarD, sigCarU,
                              K, kkH, vDk, vUk, psi, psiH)
        norm = norm + epsv1
        reldel = epsv1 / norm

    STCarD = STCarD + fD0
    STCarU = STCarU + fU0
    udifD, udifU, psi, psiH = volterra(zcel, lamCarD, lamCarU, STCarD, STCarU,
                                       K, kkH, tables=tab)
    ssD, ssU = get_intpS(zcel, p, b, SCar, w0, lamCarD, lamCarU, K, kkH,
                         udifD, udifU, psi, psiH)
    IdifD, IdifU = get_idif(zcel, p, lamCarD, lamCarU, fiD0, fiU0, ssD, ssU,
                            udifD, udifU, psi, psiH, K, kkH)

    STUsrD, STUsrU = get_fk(zcel, b_usr, Kusr, kkHusr, SUsr, w0,
                            lamCarD, lamCarU, K, kkH, udifD, udifU, psi, psiH)
    STUsrD = STUsrD + fUsrD0
    STUsrU = STUsrU + fUsrU0
    udifUsrD, udifUsrU, psiusr, psiHusr = volterra(
        zcel, lamUsrD, lamUsrU, STUsrD, STUsrU, Kusr, kkHusr, tables=tab_usr)
    ssUsrD, ssUsrU = get_intpS(zcel, p, b_usr, SUsr, w0, lamCarD, lamCarU,
                               K, kkH, udifD, udifU, psi, psiH)
    IdifUsrD, IdifUsrU = get_idif(zcel, p, lamUsrD, lamUsrU, fiUsrD0, fiUsrU0,
                                  ssUsrD, ssUsrU, udifUsrD, udifUsrU,
                                  psiusr, psiHusr, Kusr, kkHusr)
    flux_iD, flux_iU = get_h_fluxes(b, w0, IdifD, IdifU)
    Tdir = i0 * mu0 * idir[-1]
    return dict(IdifD=IdifD, IdifU=IdifU, IdifUsrD=IdifUsrD, IdifUsrU=IdifUsrU,
                flux_iD=flux_iD, flux_iU=flux_iU, Tdir=Tdir,
                udifD=udifD, udifU=udifU, ssD=ssD, ssU=ssU,
                STCarD_final=STCarD, STCarU_final=STCarU,
                udifUsrD=udifUsrD, udifUsrU=udifUsrU, reldel=reldel)


def brf_srt(H, cover, dL, ro, tau, hotspot, mu0, fi0, quad, b_usr, fi_usr,
            ro_soil, nz=20, maxiter=10, direct_fraction=1.0,
            plane_factor=1.0, alpha=0.0, tree_type=4, diam=5.0, lad=6,
            alpha_leaf=None, apf=None, apf_usr=None, apf0=None,
            apf0_usr=None, gfac=None, fi0_hot=None):
    """Total BRF at the user directions for a direct fraction ``f``.

    Mirrors the GUI/Fortran convention: the model is linear in the source, so
    a fractional direct beam is ``f*BRF(i0=1) + (1-f)*BRF(i0=0)``.
    """
    case = build_case(H, cover, dL, ro, tau, hotspot, mu0, fi0, quad,
                      b_usr, fi_usr, nz, alpha=alpha,
                      tree_type=tree_type, diam=diam, lad=lad,
                      alpha_leaf=alpha_leaf, apf=apf, apf_usr=apf_usr,
                      apf0=apf0, apf0_usr=apf0_usr, gfac=gfac,
                      fi0_hot=fi0_hot)
    s = solve_problem(case, "S", maxiter)
    s_brf = jnp.pi * s["IdifUsrU"][0]
    r_s = s["flux_iD"][-1]
    soil = 1.0 / jnp.maximum(1.0 - ro_soil * r_s, 1e-9)

    def one(problem):
        d = solve_problem(case, problem, maxiter)
        if problem == "BS_dir":
            brf_bs = jnp.pi * d["IdifUsrU"][0] / mu0
            t_bs = (d["Tdir"] + d["flux_iD"][-1]) / mu0
        else:
            brf_bs = jnp.pi * d["IdifUsrU"][0]
            t_bs = d["flux_iD"][-1]
        fac = plane_factor if problem == "BS_dir" else 1.0
        return fac * (brf_bs + ro_soil * soil * t_bs * s_brf), d, t_bs

    f = float(direct_fraction)
    if f >= 1.0 - 1e-12:
        brf, bs, t_bs = one("BS_dir")
        det = dict(bs=bs, s=s, case=case, bs_brf=jnp.pi * bs["IdifUsrU"][0] / mu0,
                   s_brf=s_brf, t_bs=t_bs, r_s=r_s)
    elif f <= 1e-12:
        brf, bd, t_bd = one("BS_dif")
        det = dict(bs_dif=bd, s=s, case=case, bs_brf=jnp.pi * bd["IdifUsrU"][0],
                   s_brf=s_brf, t_bs=t_bd, r_s=r_s)
    else:
        b1, bs, t1 = one("BS_dir")
        b0, bd, t0 = one("BS_dif")
        brf = f * b1 + (1.0 - f) * b0
        det = dict(bs=bs, bs_dif=bd, s=s, case=case, s_brf=s_brf, r_s=r_s)
    return brf, det


# --------------------------------------------------------------------------
# High-level entry point used by the GUI / inversion
# --------------------------------------------------------------------------
_QUAD_CACHE: dict = {}


def _quad_for_sza(sza):
    """Carlson quadrature (host-side, cached) for a given SZA in degrees."""
    key = round(float(sza), 4)
    if key not in _QUAD_CACHE:
        from .ref_srt import carlson_quadrature
        _QUAD_CACHE[key] = tuple(np.asarray(q, float)
                                 for q in carlson_quadrature(float(sza)))
    return _QUAD_CACHE[key]


def brf_flat_sphere(H, cover, dL, sza, saa, vza, vaa, ro, tau, ro_soil,
                    hotspot=0.0, nz=20, direct_fraction=1.0,
                    slope=0.0, aspect=90.0, tree_type=4, diam=5.0, lad=6,
                    alpha_slope=None):
    """Full SRT BRF for spherical / cylindrical / conical / ellipsoidal crowns.

    ``sza``/``saa``/``vza``/``vaa``/``slope``/``aspect`` are degrees.
    ``alpha_slope`` is the leaf-orientation rotation (Fortran AlphaSlope,
    degrees); it defaults to ``slope`` (the usual case).  LAD codes follow the
    Fortran: 1..5 planophile..uniform, 6 spherical, 7 horizontal, 8 vertical,
    9 atmosphere.
    """
    from . import lad as ladmod
    from .ref_srt import (carlson_quadrature, fortran_fi0, slope_geometry_np,
                          user_directions_slope)
    sza_f = float(sza); saa_f = float(saa)
    slope_f = float(slope); aspect_f = float(aspect)
    al_leaf = float(np.deg2rad(slope_f if alpha_slope is None else alpha_slope))
    vza = np.asarray(vza, dtype=float)
    vaa = np.asarray(vaa, dtype=float)

    # Slope rules (test.f): spherical crowns keep H and scale cover; other
    # crowns scale H (and the hot-spot parameter) and keep cover.
    cos_slope = jnp.cos(jnp.deg2rad(slope_f))
    if tree_type == 4:
        H_eff = H
        cover_eff = cover * cos_slope
        hot_eff = hotspot
    else:
        H_eff = H * cos_slope
        cover_eff = cover
        hot_eff = hotspot * cos_slope

    sza_t, saa_t, vza_t, _ = slope_geometry_np(sza_f, saa_f, vza, vaa, slope_f, aspect_f)
    b_usr, fi_usr = user_directions_slope(vza, vaa, slope_f, aspect_f)
    quad = _quad_for_sza(sza_t)
    mu0 = float(abs(np.cos(np.deg2rad(sza_t))))
    fi0_hot, fi0_inc = fortran_fi0(saa_t)
    plane_factor = mu0 / max(float(np.cos(np.deg2rad(sza_f))), 1e-9)
    alpha_rad = float(np.deg2rad(slope_f))

    b, fi, w0 = quad
    # Host-side constants for non-spherical LAD (independent of ro/tau).
    apf = apf_usr = apf0 = apf0_usr = gfac = None
    if lad in (1, 2, 3, 4, 5):
        def _pair(b1, f1, b2, f2):
            Ar, At = ladmod.apf_matrices(lad, al_leaf, b1, f1, b2, f2)
            return jnp.asarray(Ar), jnp.asarray(At)

        DD_r, DD_t = _pair(-b, fi + np.pi, -b, fi + np.pi)
        DU_r, DU_t = _pair(-b, fi + np.pi, b, fi)
        UD_r, UD_t = _pair(b, fi, -b, fi + np.pi)
        UU_r, UU_t = _pair(b, fi, b, fi)
        apf = {"DD_refl": DD_r, "DD_trans": DD_t, "DU_refl": DU_r,
               "DU_trans": DU_t, "UD_refl": UD_r, "UD_trans": UD_t,
               "UU_refl": UU_r, "UU_trans": UU_t}
        DD_r, DD_t = _pair(-b, fi + np.pi, -b_usr, fi_usr + np.pi)
        DU_r, DU_t = _pair(-b, fi + np.pi, b_usr, fi_usr)
        UD_r, UD_t = _pair(b, fi, -b_usr, fi_usr + np.pi)
        UU_r, UU_t = _pair(b, fi, b_usr, fi_usr)
        apf_usr = {"DD_refl": DD_r, "DD_trans": DD_t, "DU_refl": DU_r,
                   "DU_trans": DU_t, "UD_refl": UD_r, "UD_trans": UD_t,
                   "UU_refl": UU_r, "UU_trans": UU_t}
        D_r, D_t = _pair(np.array([-mu0]), np.array([fi0_inc]), -b, fi + np.pi)
        U_r, U_t = _pair(np.array([-mu0]), np.array([fi0_inc]), b, fi)
        apf0 = {"D_refl": D_r, "D_trans": D_t, "U_refl": U_r, "U_trans": U_t}
        D_r, D_t = _pair(np.array([-mu0]), np.array([fi0_inc]),
                         -b_usr, fi_usr + np.pi)
        U_r, U_t = _pair(np.array([-mu0]), np.array([fi0_inc]), b_usr, fi_usr)
        apf0_usr = {"D_refl": D_r, "D_trans": D_t,
                    "U_refl": U_r, "U_trans": U_t}
    gfac = {
        "dir_down": jnp.asarray(ladmod.gfun(lad, -b, fi + np.pi, al_leaf)),
        "dir_up": jnp.asarray(ladmod.gfun(lad, b, fi, al_leaf)),
        "usr_down": jnp.asarray(ladmod.gfun(lad, -b_usr, fi_usr + np.pi, al_leaf)),
        "usr_up": jnp.asarray(ladmod.gfun(lad, b_usr, fi_usr, al_leaf)),
        "sun": jnp.asarray(np.ravel(ladmod.gfun(lad, -mu0, fi0_hot, al_leaf))[0]),
    }

    brf, _ = brf_srt(H_eff, cover_eff, dL, ro, tau, hot_eff, mu0, fi0_inc, quad,
                     jnp.asarray(b_usr), jnp.asarray(fi_usr), ro_soil,
                     nz=nz, direct_fraction=direct_fraction,
                     plane_factor=plane_factor, alpha=alpha_rad,
                     tree_type=tree_type, diam=diam, lad=lad,
                     alpha_leaf=al_leaf, apf=apf, apf_usr=apf_usr,
                     apf0=apf0, apf0_usr=apf0_usr, gfac=gfac,
                     fi0_hot=fi0_hot)
    # The Fortran reports a sentinel instead of a BRF for view directions at
    # or below the local slope horizon (VZA_T >= 85 deg).
    sentinel = 0.000012345
    mask = jnp.asarray(np.asarray(vza_t) >= 85.0)
    return jnp.where(mask, sentinel, brf)

"""Exact NumPy port of the Fortran SRT / TopoSRT reference solver.

This module is the *validation oracle* for the differentiable JAX port.  It
follows ``SRT_topo`` line by line:

    EQN2PI + EQN2PI_C_k1_k2 + trans_DS_to_Plane  -> Carlson quadrature
    cond_pcf (sphere / cylinder)                  -> K, kkH
    get_sigma + get_HSF                           -> sigma, lambda
    get_Scat_coeff                                -> ScatXX matrices
    Volterra_lin                                  -> u, uH, psi, psiH
    get_f0_Fi0, get_fk_lin, get_intpS_lin, get_Idif_lin
    get_h_fluxes, norm_volterra_lin

Only the spherical-crown / flat-ground case is implemented for now (the
diagnostic case used for validation).  Cylinder/slope support is added in
the JAX port once the flat case is aligned.

Conventions (identical to the Fortran):
  * every direction is stored as an "upward" pair (b>0, fi); the
    corresponding downward direction is (-b, fi+pi);
  * K and sigma only depend on |b| for the flat case;
  * ``b`` is the polar cosine and ``fi`` the azimuth in radians.
"""

from __future__ import annotations

import numpy as np

from . import lad as ladmod

EPSX = 1e-4


# --------------------------------------------------------------------------
# Carlson quadrature
# --------------------------------------------------------------------------
def eqn2pi(nkw=18):
    """Port of EQN2PI (returns b, fi, w0, nfl)."""
    eps = 1e-4
    pih = np.pi / 2.0
    pih2 = pih / 2.0
    w = float(nkw)
    w0_scalar = 8.0 / (w * (2.0 + w))
    kmj = nkw // 2
    nfl = np.array([kmj + 1 - i for i in range(1, kmj + 1)], dtype=int)
    nfi = 4 * int(nfl.sum())
    fi = np.zeros(nfi)
    b = np.zeros(kmj)
    rm = np.zeros(kmj + 1)
    fi[0] = 0.0
    a = c = d = 0.0
    rm[0] = 4.0 / (2.0 + w)
    for i in range(kmj):
        fi[i + 1] = fi[i] + rm[i]
        b[i] = 0.5 * (fi[i] + fi[i + 1])
        rm[i + 1] = rm[i] - w0_scalar
        a += rm[i] * b[i] ** 2
        c += rm[i] * b[i] * fi[i]
        d += rm[i] * fi[i] ** 2
    a = (-c + np.sqrt(c * c - (a - 1.0 / 3.0) * d)) / d
    for i in range(kmj):
        b[i] += a * fi[i]
    c = 0.0
    for i in range(kmj):
        c += rm[i] * b[i]
    c = pih * c
    for i in range(kmj):
        rm[i] = np.sqrt(max(1.0 - b[i] ** 2, 0.0))

    l = 0
    for i in range(kmj):
        nf = nfl[i]
        for m in range(1, nf + 1):
            # Fortran: pi*(2m-1)/(w-2*i_fortran+2) with i_fortran=i+1
            fi[l + m - 1] = pih * (2.0 * m - 1.0) / (w - 2.0 * i) - pih2
        l += 4 * nf

    w0_scalar = w0_scalar * pih
    a = 1.0
    while True:
        l = 0
        d = e = 0.0
        for i in range(kmj):
            di = ei = 0.0
            nf = nfl[i]
            for m in range(1, nf + 1):
                aa = fi[l + m - 1] * a + pih2
                di += fi[l + m - 1] * np.sin(aa)
                ei += np.cos(aa)
            d += rm[i] * di
            e += rm[i] * ei
            l += 4 * nf
        au = a + (e - c / w0_scalar) / d
        if abs(au - a) < eps:
            a = au
            break
        a = au

    l = 0
    for i in range(kmj):
        nf = nfl[i]
        nfl[i] = 4 * nf
        for m in range(1, nf + 1):
            fi[l + m - 1] = fi[l + m - 1] * a + pih2
            # NB: in EQN2PI the local variable ``pi`` is acos(-1)/2, so the
            # azimuth blocks step by pi/2 (not pi).
            fi[l + m + nf - 1] = fi[l + m - 1] + pih
            fi[l + m + 2 * nf - 1] = fi[l + m + nf - 1] + pih
            fi[l + m + 3 * nf - 1] = fi[l + m + 2 * nf - 1] + pih
        l += nfl[i]
    return b, fi, w0_scalar, nfl


def eqn2pi_c_k1_k2(nkw=18, k1=4, k2=0):
    """Port of EQN2PI_C_k1_k2 (flat-format nodes: b, fi, w0, all length L)."""
    b9, fi180, w0x, nfl9 = eqn2pi(nkw)
    nfi9 = len(fi180)
    kmj9 = len(b9)
    if k1 <= 0 or k2 < 0 or (k1 > 1 and k2 != 0):
        raise ValueError("invalid k1/k2")
    if k2 == 1:
        return b9.copy(), fi180.copy(), np.full(nfi9, w0x), np.array([nfi9])
    b_out = []
    w0_out = []
    fi_out = []
    kl = 0
    for l in range(kmj9 - 1):
        nfl_l = 4 * (kmj9 - l)
        for _ in range(nfl_l):
            b_out.append(b9[l])
            w0_out.append(w0x)
        kl += nfl_l
    fi_out = list(fi180[:kl])
    theta0 = np.arccos(1.0 - 1.0 / (nfi9 / 4.0))
    step = theta0 / float(k1)
    for l in range(1, k1 + 1):
        nfl_l = 4 * (k1 - l + 1)
        psi_l = 2.0 * np.pi / float(nfl_l)
        bl = theta0 - (l - 1) * step
        eta_l1 = bl - step
        eta_l = bl if l == 1 else bl + step
        w0 = 0.5 * (np.cos(eta_l1) - np.cos(eta_l)) * psi_l
        for m in range(1, nfl_l + 1):
            b_out.append(np.cos(bl))
            w0_out.append(w0)
            fi_out.append((m - 0.5) * psi_l)
    b_out.append(1.0)
    w0_out.append(np.pi * (1.0 - b_out[-2]))
    fi_out.append(0.0)
    return (np.asarray(b_out, float), np.asarray(fi_out, float),
            np.asarray(w0_out, float), np.ones(len(b_out), dtype=int))


def trans_ds_to_plane(sza_deg, b, fi, w0):
    """Port of trans_DS_to_Plane (v05).  All nodes have nfl=1."""
    eps1 = 0.99999
    pi = np.pi
    cos_sza = np.cos(np.deg2rad(sza_deg))
    sin_sza = np.sqrt(1.0 - cos_sza ** 2)
    b_out, fi_out, w_out = [], [], []
    for l in range(len(b)):
        cos_vza = b[l]
        sin_vza = np.sqrt(max(1.0 - cos_vza ** 2, 0.0))
        omega_y = sin_vza * np.sin(fi[l])
        omega_z = cos_vza * cos_sza - sin_vza * sin_sza * np.cos(fi[l])
        if omega_z >= eps1:
            b_out.append(omega_z); fi_out.append(0.0); w_out.append(w0[l])
        elif omega_z >= 0.0:
            b_out.append(omega_z); w_out.append(w0[l])
            omega_x = cos_vza * sin_sza + sin_vza * cos_sza * np.cos(fi[l])
            ratio = omega_x / np.sqrt(max(1.0 - omega_z ** 2, 1e-30))
            if abs(ratio) > 1.0:
                ratio = ratio / abs(ratio)
            f = np.arccos(np.clip(ratio, -1.0, 1.0))
            if omega_y < 0.0:
                f = 2.0 * pi - f
            fi_out.append(f)
        omega_z = -cos_vza * cos_sza - sin_vza * sin_sza * np.cos(fi[l])
        if omega_z >= eps1:
            b_out.append(omega_z); fi_out.append(0.0); w_out.append(w0[l])
        elif omega_z >= 0.0:
            b_out.append(omega_z); w_out.append(w0[l])
            omega_x = -cos_vza * sin_sza + sin_vza * cos_sza * np.cos(fi[l])
            ratio = omega_x / np.sqrt(max(1.0 - omega_z ** 2, 1e-30))
            if abs(ratio) > 1.0:
                ratio = ratio / abs(ratio)
            f = np.arccos(np.clip(ratio, -1.0, 1.0))
            if omega_y < 0.0:
                f = 2.0 * pi - f
            fi_out.append(f)
    return np.asarray(b_out), np.asarray(fi_out), np.asarray(w_out)


def carlson_quadrature(sza_deg, nkw=18, k1=4):
    """Carlson quadrature in the plane geometry (217 nodes for nkw=18,k1=4)."""
    b, fi, w0, nfl = eqn2pi_c_k1_k2(nkw, k1, 0)
    return trans_ds_to_plane(sza_deg, b, fi, w0)


# --------------------------------------------------------------------------
# Canopy structure
# --------------------------------------------------------------------------
def p_profile(z, H, cover, tree_type=4):
    """Vegetation probability p(z) (get_probability.f)."""
    u = np.clip(np.asarray(z, float) / H, 0.0, 1.0)
    if tree_type == 1:
        return cover * np.ones_like(u)
    if tree_type == 2:
        return 1.0 - (1.0 - cover) ** (u ** 2)
    if tree_type in (3, 4):
        return 1.0 - (1.0 - cover) ** (4.0 * u * (1.0 - u))
    return cover * np.ones_like(u)


def cond_pcf_sphere(z1, z2, mu, H, cover, alpha=0.0, fi=0.0):
    """Vectorized cond_pcf case(4) (sphere), including the slope factor.

    ``alpha`` is the topographic slope (radians) used by the Fortran through
    ``par_pcf(11)``; ``fi`` is the direction azimuth.  The lam distance is
    evaluated in the form

        lam^2 = dz^2/mu^2 * [(1-mu^2) sin^2(fi)
                             + (cos(fi) cos(alpha) sqrt(1-mu^2)
                                + mu sin(alpha))^2]

    which is algebraically identical to the Fortran but avoids the
    ``1/sqrt(1-mu^2)`` singularity.
    """
    z1 = np.asarray(z1, np.float32)
    z2 = np.asarray(z2, np.float32)
    mu = np.asarray(mu, np.float32)
    Hf = np.float32(H)
    coverf = np.float32(cover)
    af = np.float32(alpha)
    ff = np.asarray(fi, np.float32)
    epsz = np.float32(1e-3) * Hf
    epsmu = np.float32(0.0017)
    one = np.float32(1.0)
    mu_abs = np.abs(mu)
    one_minus = np.maximum(one - mu * mu, np.float32(0.0))
    sq = np.sqrt(one_minus)
    # Exact Fortran operation order (real(4)), including the mu=1 branch.
    m = np.sin(ff)
    n = np.cos(ff) * np.cos(af) + mu_abs * np.sin(af) / np.maximum(sq, np.float32(1e-30))
    lam = (np.abs(z1 - z2) * sq / np.maximum(epsmu, mu_abs)
           * np.sqrt(m * m + n * n))
    lam = np.where(mu_abs == np.float32(1.0), np.float32(0.0), lam)
    lam = np.where(mu_abs == np.float32(0.0),
                   np.abs(z1 - z2) / epsmu, lam)
    rz1 = np.sqrt(np.maximum(z1 * np.abs(Hf - z1), np.float32(1e-30)))
    rz2 = np.sqrt(np.maximum(z2 * np.abs(Hf - z2), np.float32(1e-30)))
    xrz = np.minimum(rz1, rz2)
    xrs = np.maximum(rz1, rz2)

    kappa_far = one - (xrz / np.maximum(rz1, np.float32(1e-30))) ** 2
    kappa_no = np.float32(1.0)
    alfa = np.float32(2.0) * lam * xrs
    alfa = np.where(alfa > 0, (xrs * xrs - xrz * xrz + lam * lam) / np.maximum(alfa, np.float32(1e-30)), one)
    alfa = np.arccos(np.clip(alfa, np.float32(-1.0), one).astype(np.float32))
    beta = np.float32(2.0) * lam * xrz
    beta = np.where(beta > 0, (xrz * xrz - xrs * xrs + lam * lam) / np.maximum(beta, np.float32(1e-30)), one)
    beta = np.arccos(np.clip(beta, np.float32(-1.0), one).astype(np.float32))
    kappa_p = alfa * xrs * xrs + beta * xrz * xrz - lam * xrs * np.sin(alfa)
    kappa_p = kappa_p / np.maximum(np.float32(np.pi) * rz1 * rz1, np.float32(1e-30))
    kappa_p = one - kappa_p
    kappa = np.where(xrs >= xrz + lam, kappa_far,
                     np.where(lam >= xrz + xrs, kappa_no, kappa_p))

    u1 = z1 / Hf
    u2 = z2 / Hf
    pz1 = one - (one - coverf) ** (np.float32(4.0) * u1 * (one - u1))
    pz2 = one - (one - coverf) ** (np.float32(4.0) * u2 * (one - u2))
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / np.maximum(pz1, np.float32(1e-30))
    cpf = one - (one - pz2) * cpf
    z1b = (z1 <= epsz) | (np.abs(Hf - z1) <= epsz)
    z2b = (z2 <= epsz) | (np.abs(Hf - z2) <= epsz)
    cpf = np.where(z1b, np.where(lam <= rz2, one, pz2),
                   np.where(z2b, np.float32(0), cpf))
    return np.float32(cpf).astype(float)


def cond_pcf_cylinder(z1, z2, mu, H, cover, diam, alpha=0.0, fi=0.0,
                      alpha_leaf=0.0):
    """Vectorized cond_pcf case(1) (cylinder), Fortran real(4) semantics."""
    z1 = np.asarray(z1, np.float32)
    z2 = np.asarray(z2, np.float32)
    mu = np.asarray(mu, np.float32)
    Hf = np.float32(H)
    coverf = np.float32(cover)
    af = np.float32(alpha)
    ff = np.asarray(fi, np.float32)
    epsz = np.float32(1e-3) * Hf
    epsmu = np.float32(0.0017)
    one = np.float32(1.0)
    mu_abs = np.abs(mu)
    sq = np.sqrt(np.maximum(one - mu * mu, np.float32(0.0)))
    n = np.cos(ff) * np.cos(af) + mu_abs * np.sin(af) / np.maximum(sq, np.float32(1e-30))
    lam = (np.abs(z1 - z2) * sq / np.maximum(epsmu, mu_abs)
           * np.sqrt(n * n + np.sin(ff) ** 2))
    lam = np.where(mu_abs == one, np.float32(0.0), lam)
    lam = np.where(mu_abs == np.float32(0.0), np.abs(z1 - z2) / epsmu, lam)

    R = np.float32(diam) / np.float32(2.0)
    rz1 = np.full_like(z1, R)
    rz2 = np.full_like(z2, R)
    xrz = np.minimum(rz1, rz2)
    xrs = np.maximum(rz1, rz2)
    Rone = one
    Rtwo = rz2 / rz1
    alf = np.float32(alpha_leaf)
    dd = np.sqrt(np.maximum(one - np.cos(ff) ** 2 * np.sin(alf) ** 2, np.float32(0.0)))
    dd = dd * lam / rz1
    alfa = np.float32(2.0) * dd * Rone
    alfa = (dd * dd + Rone * Rone - Rtwo * Rtwo) / np.maximum(alfa, np.float32(1e-30))
    alfa = np.arccos(np.clip(alfa, np.float32(-1.0), one).astype(np.float32))
    beta = np.float32(2.0) * dd * Rtwo
    beta = (dd * dd + Rtwo * Rtwo - Rone * Rone) / np.maximum(beta, np.float32(1e-30))
    beta = np.arccos(np.clip(beta, np.float32(-1.0), one).astype(np.float32))
    kp = (Rone + Rtwo - dd) * (dd + Rone - Rtwo) * (dd - Rone + Rtwo) * (dd + Rone + Rtwo)
    kp = np.float32(0.5) * np.sqrt(np.maximum(kp, np.float32(0.0)))
    kp = Rone * Rone * alfa + Rtwo * Rtwo * beta - kp
    kp = kp / np.float32(np.pi)
    kp = one - kp
    kappa = np.where(dd >= Rone + Rtwo, one,
                     np.where(dd <= np.abs(Rone - Rtwo),
                              one - np.minimum(Rone * Rone, Rtwo * Rtwo), kp))

    u1 = z1 / Hf
    u2 = z2 / Hf
    pz1 = coverf + np.zeros_like(u1)
    pz2 = coverf + np.zeros_like(u2)
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / np.maximum(pz1, np.float32(1e-30))
    cpf = one - (one - pz2) * cpf
    # case(1) only checks the upper boundary (no |H-z| test)
    cpf = np.where(z1 <= epsz, np.where(lam <= rz2, one, pz2),
                   np.where(z2 <= epsz, np.float32(0), cpf))
    return np.float32(cpf).astype(float)


def cond_pcf_cone(z1, z2, mu, H, cover, diam, alpha=0.0, fi=0.0,
                  alpha_leaf=0.0):
    """cond_pcf case(2) (conical crowns)."""
    z1 = np.asarray(z1, np.float32); z2 = np.asarray(z2, np.float32)
    mu = np.asarray(mu, np.float32); Hf = np.float32(H)
    coverf = np.float32(cover)
    af = np.float32(alpha)
    alf = np.float32(alpha_leaf)
    ff = np.asarray(fi, np.float32)
    epsz = np.float32(1e-3) * Hf
    epsmu = np.float32(0.0017)
    one = np.float32(1.0)
    mu_abs = np.abs(mu)
    sq = np.sqrt(np.maximum(one - mu * mu, np.float32(0.0)))
    n = np.cos(ff) * np.cos(af) + mu_abs * np.sin(af) / np.maximum(sq, np.float32(1e-30))
    lam = (np.abs(z1 - z2) * sq / np.maximum(epsmu, mu_abs)
           * np.sqrt(n * n + np.sin(ff) ** 2))
    lam = np.where(mu_abs == one, np.float32(0.0), lam)
    lam = np.where(mu_abs == np.float32(0.0), np.abs(z1 - z2) / epsmu, lam)

    R = np.float32(diam) / np.float32(2.0)
    cosA = np.cos(alf)
    rh = R / Hf * cosA
    rz1 = rh * z1 / cosA
    rz2 = rh * z2 / cosA
    Rone = one
    Rtwo = rz2 / np.maximum(rz1, np.float32(1e-30))
    dd = np.sqrt(np.maximum(one - np.cos(ff) ** 2 * np.sin(alf) ** 2, np.float32(0.0)))
    dd = dd * lam / np.maximum(rz1, np.float32(1e-30))
    alfa = np.float32(2.0) * dd * Rone
    alfa = (dd * dd + Rone * Rone - Rtwo * Rtwo) / np.maximum(alfa, np.float32(1e-30))
    alfa = np.arccos(np.clip(alfa, np.float32(-1.0), one).astype(np.float32))
    beta = np.float32(2.0) * dd * Rtwo
    beta = (dd * dd + Rtwo * Rtwo - Rone * Rone) / np.maximum(beta, np.float32(1e-30))
    beta = np.arccos(np.clip(beta, np.float32(-1.0), one).astype(np.float32))
    kp = (Rone + Rtwo - dd) * (dd + Rone - Rtwo) * (dd - Rone + Rtwo) * (dd + Rone + Rtwo)
    kp = np.float32(0.5) * np.sqrt(np.maximum(kp, np.float32(0.0)))
    kp = Rone * Rone * alfa + Rtwo * Rtwo * beta - kp
    kp = kp / np.float32(np.pi)
    kp = one - kp
    kappa = np.where(dd >= Rone + Rtwo, one,
                     np.where(dd <= np.abs(Rone - Rtwo),
                              one - np.minimum(Rone * Rone, Rtwo * Rtwo), kp))

    u1 = z1 / Hf; u2 = z2 / Hf
    gap = one - coverf
    pz1 = one - gap ** (u1 * u1)
    pz2 = one - gap ** (u2 * u2)
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / np.maximum(pz1, np.float32(1e-30))
    cpf = one - (one - pz2) * cpf
    # case(2) only checks the upper boundary (no |H-z| test)
    cpf = np.where(z1 <= epsz, np.where(lam <= rz2, one, pz2),
                   np.where(z2 <= epsz, np.float32(0), cpf))
    return np.float32(cpf).astype(float)


def cond_pcf_ellipsoid(z1, z2, mu, H, cover, diam, alpha=0.0, fi=0.0,
                       alpha_leaf=0.0):
    """cond_pcf case(3) (ellipsoidal crowns)."""
    z1 = np.asarray(z1, np.float32); z2 = np.asarray(z2, np.float32)
    mu = np.asarray(mu, np.float32); Hf = np.float32(H)
    coverf = np.float32(cover)
    af = np.float32(alpha)
    alf = np.float32(alpha_leaf)
    ff = np.asarray(fi, np.float32)
    epsz = np.float32(1e-3) * Hf
    epsmu = np.float32(0.0017)
    one = np.float32(1.0)
    mu_abs = np.abs(mu)
    sq = np.sqrt(np.maximum(one - mu * mu, np.float32(0.0)))
    n = np.cos(ff) * np.cos(af) + mu_abs * np.sin(af) / np.maximum(sq, np.float32(1e-30))
    lam = (np.abs(z1 - z2) * sq / np.maximum(epsmu, mu_abs)
           * np.sqrt(n * n + np.sin(ff) ** 2))
    lam = np.where(mu_abs == one, np.float32(0.0), lam)
    lam = np.where(mu_abs == np.float32(0.0), np.abs(z1 - z2) / epsmu, lam)

    cosA = np.cos(alf)
    aratio = np.float32(diam) / Hf * cosA  # Fortran: par_pcf(1)/h (diameter)
    rz1 = z1 / cosA
    rz1 = rz1 * np.abs(Hf / cosA - z1 / cosA)
    rz1 = aratio * np.sqrt(np.maximum(rz1, np.float32(0.0)))
    rz2 = z2 / cosA
    rz2 = rz2 * np.abs(Hf / cosA - z2 / cosA)
    rz2 = aratio * np.sqrt(np.maximum(rz2, np.float32(0.0)))
    Rone = one
    Rtwo = rz2 / np.maximum(rz1, np.float32(1e-30))
    dd = np.sqrt(np.maximum(one - np.cos(ff) ** 2 * np.sin(alf) ** 2, np.float32(0.0)))
    dd = dd * lam / np.maximum(rz1, np.float32(1e-30))
    alfa = np.float32(2.0) * dd * Rone
    alfa = (dd * dd + Rone * Rone - Rtwo * Rtwo) / np.maximum(alfa, np.float32(1e-30))
    alfa = np.arccos(np.clip(alfa, np.float32(-1.0), one).astype(np.float32))
    beta = np.float32(2.0) * dd * Rtwo
    beta = (dd * dd + Rtwo * Rtwo - Rone * Rone) / np.maximum(beta, np.float32(1e-30))
    beta = np.arccos(np.clip(beta, np.float32(-1.0), one).astype(np.float32))
    kp = (Rone + Rtwo - dd) * (dd + Rone - Rtwo) * (dd - Rone + Rtwo) * (dd + Rone + Rtwo)
    kp = np.float32(0.5) * np.sqrt(np.maximum(kp, np.float32(0.0)))
    kp = Rone * Rone * alfa + Rtwo * Rtwo * beta - kp
    kp = kp / np.float32(np.pi)
    kp = one - kp
    kappa = np.where(dd >= Rone + Rtwo, one,
                     np.where(dd <= np.abs(Rone - Rtwo),
                              one - np.minimum(Rone * Rone, Rtwo * Rtwo), kp))

    u1 = z1 / Hf; u2 = z2 / Hf
    gap = one - coverf
    pz1 = one - gap ** (4.0 * u1 * (one - u1))
    pz2 = one - gap ** (4.0 * u2 * (one - u2))
    cpf = (one - pz1) ** kappa
    cpf = (one - cpf) / np.maximum(pz1, np.float32(1e-30))
    cpf = one - (one - pz2) * cpf
    z1b = (z1 <= epsz) | (np.abs(Hf - z1) <= epsz)
    z2b = (z2 <= epsz) | (np.abs(Hf - z2) <= epsz)
    cpf = np.where(z1b, np.where(lam <= rz2, one, pz2),
                   np.where(z2b, np.float32(0), cpf))
    return np.float32(cpf).astype(float)


def cond_pcf(z1, z2, mu, H, cover, diam, tree_type=4, alpha=0.0, fi=0.0,
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
    """Build K (nfi,nz,nz) and kkH (nfi,nz) exactly as get_kernel.f."""
    zcel = np.asarray(zcel, float)
    nz = len(zcel) - 1
    b = np.asarray(b, float)
    nfi = len(b)
    if fi is None:
        fi = np.zeros(nfi)
    fi = np.asarray(fi, float)
    K = np.zeros((nfi, nz, nz))
    KH = np.zeros((nfi, nz))
    zi = zcel[:-1]
    zip1 = zcel[1:]
    for m in range(nfi):
        mu = b[m]
        f = np.array([fi[m]])
        ki = cond_pcf(zi, zi, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
        ki1 = cond_pcf(zip1, zi, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
        ki2 = cond_pcf(zip1, zip1, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
        ki3 = cond_pcf(zi, zip1, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
        KH[m] = (ki + ki2 + ki3) / 3.0
        idx = np.arange(nz)
        diag = (ki + ki1 + ki2) / 3.0
        for i in range(nz):
            v1 = cond_pcf(zi[i], zi, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
            v2 = cond_pcf(zip1[i], zi, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
            v3 = cond_pcf(zip1[i], zip1, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
            v4 = cond_pcf(zi[i], zip1, mu, H, cover, diam, tree_type, alpha, f, alpha_leaf)
            K[m, i, :] = (v1 + v2 + v3 + v4) / 4.0
        K[m, idx, idx] = diag
    return K, KH


def kernel_sun(zcel, mu0, H, cover, fi0=0.0, alpha=0.0, diam=5.0, tree_type=4,
               alpha_leaf=None):
    """K in the solar direction (single direction)."""
    K, KH = kernel_arrays(zcel, [mu0], H, cover, fi=[fi0], alpha=alpha,
                          diam=diam, tree_type=tree_type, alpha_leaf=alpha_leaf)
    return K[0], KH[0]


# --------------------------------------------------------------------------
# Phase functions / scattering matrices
# --------------------------------------------------------------------------
def gind_sphere(b1, f1, b2, f2, ro, tau):
    s1 = np.sqrt(np.maximum(1.0 - b1 * b1, 0.0))
    s2 = np.sqrt(np.maximum(1.0 - b2 * b2, 0.0))
    cb = np.clip(b1 * b2 + s1 * s2 * np.cos(f2 - f1), -1.0, 1.0)
    ang = np.arccos(cb)
    sb = np.sqrt(np.maximum(1.0 - cb * cb, 0.0))
    wleaf = ro + tau
    return (wleaf * (sb - ang * cb) / (3.0 * np.pi) + tau * cb / 3.0) / np.pi


def scatter_matrices(bc, fic, bu, fiu, ro, tau, dL, lad=6, alpha_slope=0.0):
    """(nfi_c, nfi_out) matrices for DD, DU, UD, UU (see get_Scat_coeff)."""
    BC = np.asarray(bc)[:, None]
    FC = np.asarray(fic)[:, None]
    BD = np.asarray(bu)[None, :]
    FD = np.asarray(fiu)[None, :]
    dpi = np.pi
    if lad in (6, 7, 8, 9):
        DD = dL * ladmod.gind_closed(lad, -BC, FC + dpi, -BD, FD + dpi, ro, tau)
        DU = dL * ladmod.gind_closed(lad, -BC, FC + dpi, BD, FD, ro, tau)
        UD = dL * ladmod.gind_closed(lad, BC, FC, -BD, FD + dpi, ro, tau)
        UU = dL * ladmod.gind_closed(lad, BC, FC, BD, FD, ro, tau)
    else:
        Ar, At = ladmod.apf_matrices(lad, alpha_slope, -np.ravel(BC),
                                     np.ravel(FC) + dpi,
                                     -np.ravel(BD), np.ravel(FD) + dpi)
        DD = dL * (ro * Ar + tau * At)
        Ar, At = ladmod.apf_matrices(lad, alpha_slope, -np.ravel(BC),
                                     np.ravel(FC) + dpi,
                                     np.ravel(BD), np.ravel(FD))
        DU = dL * (ro * Ar + tau * At)
        Ar, At = ladmod.apf_matrices(lad, alpha_slope, np.ravel(BC), np.ravel(FC),
                                     -np.ravel(BD), np.ravel(FD) + dpi)
        UD = dL * (ro * Ar + tau * At)
        Ar, At = ladmod.apf_matrices(lad, alpha_slope, np.ravel(BC), np.ravel(FC),
                                     np.ravel(BD), np.ravel(FD))
        UU = dL * (ro * Ar + tau * At)
    return DD, DU, UD, UU


def scatter_sun(bc, fic, mu0, fi0, ro, tau, dL, lad=6, alpha_slope=0.0):
    """Solar direction -> directions.  fi0 is the physical solar azimuth
    (already including the +pi used by the Fortran: fi0 = fi0_deg*pi/180+pi)."""
    BC = np.asarray(bc)
    if lad in (6, 7, 8, 9):
        SCar0D = dL * ladmod.gind_closed(lad, -mu0, fi0, -BC, fic + np.pi, ro, tau)
        SCar0U = dL * ladmod.gind_closed(lad, -mu0, fi0, BC, fic, ro, tau)
    else:
        Ar, At = ladmod.apf_matrices(lad, alpha_slope, np.atleast_1d(-mu0),
                                     np.atleast_1d(fi0), -BC, fic + np.pi)
        SCar0D = dL * (ro * Ar[0] + tau * At[0])
        Ar, At = ladmod.apf_matrices(lad, alpha_slope, np.atleast_1d(-mu0),
                                     np.atleast_1d(fi0), BC, fic)
        SCar0U = dL * (ro * Ar[0] + tau * At[0])
    return SCar0D, SCar0U


def gfun_spherical(*args, **kwargs):
    return 0.5


def hotspot_factor(hotspot, dL, mu0d, fi0d, mu, fi, sig0, sig):
    """Port of get_HSF (Kuusk hot-spot factor)."""
    eps = 0.01745
    epsx = 1e-4
    mu = np.asarray(mu, float)
    fi = np.asarray(fi, float)
    hsf = np.ones_like(mu)
    if hotspot is None or hotspot <= 0:
        return hsf
    mask = mu > eps
    if not np.any(mask):
        return hsf
    m = mu[mask]
    f = fi[mask]
    sig_m = np.asarray(sig)[mask] if np.ndim(sig) else sig
    cosgamma = mu0d * m + np.sqrt(np.maximum((1.0 - mu0d ** 2) * (1.0 - m ** 2), 0.0)) * np.cos(f - fi0d)
    delta = mu0d ** 2 + m ** 2 + 2.0 * cosgamma * np.abs(m * mu0d)
    delta = np.sqrt(np.abs(delta)) / np.abs(m * mu0d)
    x = hotspot * delta
    x = np.where(x <= epsx, 1.0, np.where(x > 0, (1.0 - np.exp(-x)) / np.maximum(x, 1e-30), 1.0))
    fac = np.sqrt(np.abs(sig0 * m) / np.abs(sig_m * mu0d))
    hsf[mask] = np.abs(1.0 - fac * x)
    return hsf


# --------------------------------------------------------------------------
# Volterra solver and source terms
# --------------------------------------------------------------------------
def _ratio_from_xx(xx):
    """ratio = (1-exp(-xx))/xx, with the epsx branch of the Fortran."""
    small = xx <= EPSX
    x = np.where(small, 1.0, xx)
    return np.where(small, 1.0, (1.0 - np.exp(-x)) / x)


def _eta_from_xx(xx):
    """eta = 1 - (1-exp(-xx))/xx, with the epsx branch of the Fortran."""
    return 1.0 - _ratio_from_xx(xx)


def volterra(zcel, lamD, lamU, f, fH, K, kkH):
    """Port of Volterra_lin_MK_zero_v02.f.

    lamD/lamU: (nfi,), f/fH: (nz+1,nfi), K: (nfi,nz,nz), kkH: (nfi,nz).

    The inner loop over j is evaluated as a masked reduction (same algebra:
    psi(i) = f(i+1) - sum_{j<i} kij*(...)), keeping the JAX graph small.
    """
    nz = len(zcel) - 1
    nfi = lamD.shape[0]
    h = np.diff(zcel)
    u = np.zeros((nz + 1, nfi))
    uH = np.zeros((nz + 1, nfi))
    psi = np.zeros((nz, nfi))
    psiH = np.zeros((nz, nfi))
    u[0] = f[0]
    uH[nz] = fH[nz]
    Kd = np.einsum("mii->mi", K)          # (nfi,nz)
    j0 = np.arange(nz)
    jh0 = nz - 1 - j0
    mask_lt = (j0[:, None] < j0[None, :])  # mask_lt[j,i] = j < i
    kD = K / Kd[:, None, :]                # kD[d,i,j] = K[i,j]/K[j,j]
    Kr = K[:, ::-1, ::-1]
    # Volterra kij1 = K[i1,j1]/kkH[j1]  (denominator is kkH, not K[j,j])
    kU = Kr / kkH[:, ::-1][:, None, :]
    for i in range(1, nz + 1):
        i1 = nz - i
        psi[i - 1] = f[i]
        psiH[i1] = fH[i1]
        zargD = (lamD[:, None] * Kd) * h[None, :]
        A = (u[:-1].T * (1.0 - np.exp(-zargD))
             + (psi.T - u[:-1].T) * _eta_from_xx(zargD))
        contrib = np.sum(np.where(mask_lt[:, i - 1][None, :], kD[:, i - 1, :] * A, 0.0), axis=1)
        psi[i - 1] = psi[i - 1] - contrib
        zargU = (lamU[:, None] * kkH[:, jh0]) * h[jh0][None, :]
        BU = (uH[jh0 + 1].T * (1.0 - np.exp(-zargU))
              + (psiH[jh0].T - uH[jh0 + 1].T) * _eta_from_xx(zargU))
        contribU = np.sum(np.where(mask_lt[:, i - 1][None, :], kU[:, i - 1, :] * BU, 0.0), axis=1)
        psiH[i1] = psiH[i1] - contribU
        xx = lamD * Kd[:, i - 1] * h[i - 1]
        unew = u[i - 1] * np.exp(-xx) + (psi[i - 1] - u[i - 1]) * _ratio_from_xx(xx)
        neg = unew < 0.0
        psi[i - 1] = np.where(neg, u[i - 1], psi[i - 1])
        u[i] = np.where(neg, u[i - 1] * np.exp(-xx), unew)
        xx = lamU * kkH[:, i1] * h[i1]
        uHnew = uH[i1 + 1] * np.exp(-xx) + (psiH[i1] - uH[i1 + 1]) * _ratio_from_xx(xx)
        neg = uHnew < 0.0
        psiH[i1] = np.where(neg, uH[i1 + 1], psiH[i1])
        uH[i1] = np.where(neg, uH[i1 + 1] * np.exp(-xx), uHnew)
    return u, uH, psi, psiH


def get_idir(zcel, p, lam0, Kd0, psi0, udir):
    nz = len(zcel) - 1
    h = np.diff(zcel)
    idir = np.zeros(nz + 1)
    idir[0] = 1.0
    xxpsi = 0.0
    for i in range(1, nz + 1):
        pmean = 0.5 * (p[i - 1] + p[i])
        zarg = lam0 * Kd0[i - 1] * h[i - 1]
        eta = 1.0 - (1.0 - np.exp(-zarg)) / zarg
        eta = udir[i - 1] * (1.0 - np.exp(-zarg)) + (psi0[i - 1] - udir[i - 1]) * eta
        xxpsi += pmean * eta / Kd0[i - 1]
        idir[i] = 1.0 - xxpsi
    return idir


def get_f0_fi0(zcel, p, K, kkH, lam0, SCar0D, SCar0U, udir, idir,
               psi0, d0, dH, i0, b):
    """Port of get_f0_Fi0_v02.f (vectorized over layers)."""
    nz = len(zcel) - 1
    nfi = b.shape[0]
    h = np.diff(zcel)
    lam0bl = lam0 * b
    xxD = i0 * SCar0D / lam0bl
    xxU = i0 * SCar0U / lam0bl
    Kd = np.einsum("mii->mi", K)          # (nfi,nz)
    idx = np.arange(nz)
    j0 = np.arange(nz)
    jh0 = nz - 1 - j0
    mask_le = (j0[:, None] <= j0[None, :])   # [j,i] = j <= i
    zargD = (lam0 * Kd) * h[None, :]
    etaD = (udir[None, :-1] * (1.0 - np.exp(-zargD))
            + (psi0[None, :] - udir[None, :-1]) * _eta_from_xx(zargD))   # (nfi,nz)
    zargU = (lam0 * Kd[:, jh0]) * h[jh0][None, :]
    etaU = (udir[None, jh0] * (1.0 - np.exp(-zargU))
            + (psi0[None, jh0] - udir[None, jh0]) * _eta_from_xx(zargU))
    kD = K / Kd[:, None, :]
    fD0i = np.sum(np.where(mask_le.T[None], kD * etaD[:, None, :], 0.0), axis=2)
    Kr = K[:, ::-1, ::-1]
    kU = Kr / Kd[:, ::-1][:, None, :]
    kU[:, idx, idx] = kkH[:, ::-1] / Kd[:, ::-1]
    fU0i = np.sum(np.where(mask_le.T[None], kU * etaU[:, None, :], 0.0), axis=2)
    fD0i = fD0i.T   # (nz,nfi) indexed by i0=i-1
    fU0i = fU0i.T
    fiD0 = np.zeros((nz + 1, nfi))
    fiU0 = np.zeros((nz + 1, nfi))
    fD0 = np.zeros((nz + 1, nfi))
    fU0 = np.zeros((nz + 1, nfi))
    fiD0[0] = d0; fiU0[nz] = dH; fD0[0] = d0; fU0[nz] = dH
    for i in range(1, nz + 1):
        fiD0[i] = xxD * (1.0 - idir[i]) + d0
        fiU0[i - 1] = xxU * (idir[i - 1] - idir[nz]) + dH
        fD0[i] = fD0i[i - 1] * xxD + d0
        fU0[nz - i] = fU0i[i - 1] * xxU + dH
    return fD0, fU0, fiD0, fiU0


def layer_etas(zcel, u, uH, psi, psiH, lamD, lamU, K, kkH):
    """etaD/etaU per layer for the scattering-source integrals.

    Returns (etaD, etaU) with shape (nz, nfi).  Matches the eta expressions in
    get_fk_lin / get_intpS_lin (they include the division by lam*K).
    """
    nz = len(zcel) - 1
    h = np.diff(zcel)
    Kd = np.einsum("mii->mi", K)  # (nfi,nz)
    zargD = (lamD * Kd.T) * h[:, None]          # (nz,nfi)
    ratioD = np.where(zargD <= EPSX, 1.0,
                      (1.0 - np.exp(-np.where(zargD <= EPSX, 1.0, zargD)))
                      / np.where(zargD <= EPSX, 1.0, zargD))
    etaD = np.where(
        zargD <= EPSX,
        0.5 * (u[:-1] + psi) * h[:, None],
        (u[:-1] * (1.0 - np.exp(-zargD))
         + (psi - u[:-1]) * (1.0 - ratioD)) / np.maximum(lamD * Kd.T, 1e-30),
    )
    zargU = (lamU * kkH.T) * h[:, None]
    ratioU = np.where(zargU <= EPSX, 1.0,
                      (1.0 - np.exp(-np.where(zargU <= EPSX, 1.0, zargU)))
                      / np.where(zargU <= EPSX, 1.0, zargU))
    etaU = np.where(
        zargU <= EPSX,
        0.5 * (uH[1:] + psiH) * h[:, None],
        (uH[1:] * (1.0 - np.exp(-zargU))
         + (psiH - uH[1:]) * (1.0 - ratioU)) / np.maximum(lamU * kkH.T, 1e-30),
    )
    return etaD, etaU


def _xx_from_etas(etaD, etaU, w0, SDD, SDU, SUU, SUD):
    """XXD/XXU of get_fk_lin/get_intpS_lin for all layers."""
    eD = etaD * w0[None, :]
    eU = etaU * w0[None, :]
    xxD = eU @ SUD + eD @ SDD
    xxU = eU @ SUU + eD @ SDU
    return xxD, xxU


def get_fk(zcel, b_out, K_out, kkH_out, Scat, w0,
          lam1D, lam1U, K1, kkH1, u1, uH1, psi1, psiH1):
    """Port of get_fk_lin_v02A.f.

    Scat = (SDD, SDU, SUU, SUD).  K_out/kkH_out are for the outgoing
    directions; K1/kkH1 and u1/... are the Carlson fields.
    """
    nz = len(zcel) - 1
    nfi_out = b_out.shape[0]
    SDD, SDU, SUD, SUU = Scat  # scatter_matrices order: DD, DU, UD, UU
    etaD, etaU = layer_etas(zcel, u1, uH1, psi1, psiH1, lam1D, lam1U, K1, kkH1)
    xxD, xxU = _xx_from_etas(etaD, etaU, w0, SDD, SDU, SUU, SUD)  # (nz,nfi_out)

    Kd_out = np.einsum("mii->mi", K_out)
    tril = np.tril(np.ones((nz, nz)))
    fDk = np.einsum("mij,jm->im", K_out * tril[None], xxD) / b_out[None, :]
    Kup = K_out.copy()
    idx = np.arange(nz)
    Kup[:, idx, idx] = kkH_out
    triu = np.triu(np.ones((nz, nz)))
    fUk = np.einsum("mrc,cm->rm", Kup * triu[None], xxU) / b_out[None, :]
    fDk = np.vstack([np.zeros((1, nfi_out)), fDk])
    fUk = np.vstack([fUk, np.zeros((1, nfi_out))])
    return fDk, fUk


def get_intpS(zcel, p, b_out, Scat, w0, lam1D, lam1U, K1, kkH1,
              u1, uH1, psi1, psiH1):
    """Port of get_intpS_lin_v02A.f: ssD/ssU in the outgoing directions."""
    nz = len(zcel) - 1
    nfi_out = b_out.shape[0]
    SDD, SDU, SUD, SUU = Scat  # scatter_matrices order: DD, DU, UD, UU
    etaD, etaU = layer_etas(zcel, u1, uH1, psi1, psiH1, lam1D, lam1U, K1, kkH1)
    xxD, xxU = _xx_from_etas(etaD, etaU, w0, SDD, SDU, SUU, SUD)
    pmean = 0.5 * (p[:-1] + p[1:])
    ssD = np.vstack([np.zeros((1, nfi_out)),
                     np.cumsum(pmean[:, None] * xxD, axis=0)]) / b_out[None, :]
    termU = pmean[:, None] * xxU
    rev = np.flipud(np.cumsum(np.flipud(termU), axis=0)) / b_out[None, :]
    ssU = np.vstack([rev, np.zeros((1, nfi_out))])
    return ssD, ssU


def get_idif(zcel, p, lamD, lamU, fiD0, fiU0, ssD, ssU, u, uH, psi, psiH,
             K, kkH):
    """Port of get_Idif_lin_V02A.f."""
    nz = len(zcel) - 1
    h = np.diff(zcel)
    nfi = u.shape[1]
    idifD = np.zeros((nz + 1, nfi))
    idifU = np.zeros((nz + 1, nfi))
    idifD[0] = fiD0[0]
    idifU[nz] = fiU0[nz]
    Kd = np.einsum("mii->mi", K)
    xxpsi = np.zeros(nfi)
    xxpsih = np.zeros(nfi)
    for i in range(1, nz + 1):
        pmean = 0.5 * (p[i - 1] + p[i])
        zarg = lamD * Kd[:, i - 1] * h[i - 1]
        eta = _eta_from_xx(zarg)
        eta = u[i - 1] * (1.0 - np.exp(-zarg)) + (psi[i - 1] - u[i - 1]) * eta
        eta = eta / Kd[:, i - 1]
        xxpsi = xxpsi + pmean * eta
        idifD[i] = fiD0[i] + ssD[i] - xxpsi
        ih = nz - i
        pmean = 0.5 * (p[ih] + p[ih + 1])
        zarg = lamU * kkH[:, ih] * h[ih]
        eta = _eta_from_xx(zarg)
        eta = uH[ih + 1] * (1.0 - np.exp(-zarg)) + (psiH[ih] - uH[ih + 1]) * eta
        eta = eta / kkH[:, ih]
        xxpsih = xxpsih + pmean * eta
        idifU[ih] = fiU0[ih] + ssU[ih] - xxpsih
    return idifD, idifU


def get_h_fluxes(b, w0, IdifD, IdifU):
    fluxD = np.sum(b[None, :] * w0[None, :] * IdifD, axis=1)
    fluxU = np.sum(b[None, :] * w0[None, :] * IdifU, axis=1)
    return fluxD, fluxU


def norm_volterra(zcel, p, w0, lamD, lamU, sigD, sigU, K, kkH,
                  u, uH, psi, psiH):
    """Port of norm_volterra_lin_v02A.f."""
    nz = len(zcel) - 1
    h = np.diff(zcel)
    Kd = np.einsum("mii->mi", K)
    norm = 0.0
    for i in range(1, nz + 1):
        zargD = lamD * Kd[:, i - 1] * h[i - 1]
        xxD = 1.0 - _eta_from_xx(zargD)
        xxD = u[i - 1] * (1.0 - np.exp(-zargD)) + (psi[i - 1] - u[i - 1]) * xxD
        useD = sigD < lamD
        xxD = np.where(useD, sigD * xxD / np.maximum(lamD * Kd[:, i - 1], 1e-30), 0.0)
        zargU = lamU * kkH[:, i - 1] * h[i - 1]
        xxU = 1.0 - _eta_from_xx(zargU)
        xxU = uH[i] * (1.0 - np.exp(-zargU)) + (psiH[i - 1] - uH[i]) * xxU
        useU = sigU < lamU
        xxU = np.where(useU, sigU * xxU / np.maximum(lamU * kkH[:, i - 1], 1e-30), 0.0)
        xx = np.sum((xxU + xxD) * w0)
        pmean = 0.5 * (p[i - 1] + p[i])
        norm += pmean * xx
    return norm




# --------------------------------------------------------------------------
# Slope geometry (NumPy twin of diff_srt.topography)
# --------------------------------------------------------------------------
def plane_to_slope(theta, phi, slope, aspect):
    """Plane -> slope frame, single-precision twin of test.f (radians in)."""
    f32 = np.float32
    alpha, beta = f32(slope), f32(aspect)
    th, ph = f32(theta), f32(phi)
    pi = f32(np.pi)
    st, ct = f32(np.sin(th)), f32(np.cos(th))
    dphi = ph - beta
    x = f32(np.cos(dphi) * st * np.cos(alpha) - ct * np.sin(alpha))
    y = f32(np.sin(f32(beta - ph)) * st)
    z = f32(np.cos(dphi) * st * np.sin(alpha) + ct * np.cos(alpha))
    th_s = f32(np.arccos(np.clip(z, f32(-1.0), f32(1.0))))
    ph_s = f32(f32(pi / f32(2.0)) - np.arctan2(y, x))
    ph_s = np.where(ph_s < 0, ph_s + f32(2.0) * pi, ph_s)
    return np.asarray(th_s, float), np.asarray(ph_s, float)


def slope_geometry_np(sza, saa, vza, vaa, slope, aspect):
    sza_s, saa_s = plane_to_slope(np.deg2rad(sza), np.deg2rad(saa),
                                  np.deg2rad(slope), np.deg2rad(aspect))
    vza_s, vaa_s = plane_to_slope(np.deg2rad(np.asarray(vza, float)),
                                  np.deg2rad(np.asarray(vaa, float)),
                                  np.deg2rad(slope), np.deg2rad(aspect))
    return np.rad2deg(sza_s), np.rad2deg(saa_s), np.rad2deg(vza_s), np.rad2deg(vaa_s)


def fortran_fi0(saa_deg):
    """Fortran's real(4) solar azimuth roundtrip: (fi0 for get_HSF, incident)."""
    pi32 = np.float32(np.pi)
    deg = np.float32(saa_deg)
    fi0_32 = np.float32(deg * pi32 / np.float32(180.0) + pi32)
    fi_inc = np.float32(np.float32(fi0_32 - pi32) + pi32)
    return float(fi0_32), float(fi_inc)


def user_directions_slope(vza_deg, vaa_deg, slope, aspect):
    """User directions in the slope frame, mirroring test.f's real(4) formulas.

    b = cos(slope) cos(VZA) + sin(slope) sin(VZA) cos(aspect - VAA)
    """
    f32 = np.float32
    alpha = f32(np.deg2rad(slope))
    beta = f32(np.deg2rad(aspect))
    vza = np.deg2rad(np.asarray(vza_deg, np.float32))
    vaa = np.deg2rad(np.asarray(vaa_deg, np.float32))
    b = f32(np.cos(alpha) * np.cos(vza)
            + np.sin(alpha) * np.sin(vza) * np.cos(beta - vaa))
    b = np.clip(b, f32(-1.0), f32(1.0))
    xp = f32(np.sin(vza) * np.cos(alpha) * np.cos(vaa - beta)
             - np.cos(vza) * np.sin(alpha))
    yp = f32(np.sin(beta - vaa) * np.sin(vza))
    fi = np.arctan2(yp, xp)
    fi = f32(f32(np.pi / f32(2.0)) - fi)
    fi = np.where(fi < 0, fi + f32(2.0) * f32(np.pi), fi)
    fi = np.where(fi >= f32(2.0) * f32(np.pi), fi - f32(2.0) * f32(np.pi), fi)
    return b.astype(float), fi.astype(float)


def effective_canopy_np(H, cover, hotspot, slope, tree_type=4):
    """Manuscript slope rules (applied once, before the rounds)."""
    cos_slope = np.cos(np.deg2rad(slope))
    if tree_type == 4:
        return H, cover * cos_slope, hotspot
    return H * cos_slope, cover, hotspot * cos_slope


# --------------------------------------------------------------------------
# Full problem driver
# --------------------------------------------------------------------------
def user_directions(vza_deg, vaa_deg):
    b_usr = np.cos(np.deg2rad(np.asarray(vza_deg, float)))
    fi_usr = np.deg2rad(np.asarray(vaa_deg, float))
    return b_usr, fi_usr


def run_problem(H=5.0, cover=0.49397, dL=1.0, ro=0.4746, tau=0.4843,
                hotspot=8.0, sza=45.0, vaa=270.0, nz=20, maxiter=10,
                problem="BS_dir", vza_usr=(0, 15, 30, 45, 60),
                vaa_usr=(0, 90, 180, 270), quad=None, verbose=False,
                slope=0.0, aspect=90.0, tree_type=4, diam=5.0, lad=6,
                alpha_slope=None):
    """Run one SRT sub-problem; returns a dict with all fields."""
    vza = np.asarray(vza_usr, float)
    vaa_ = np.asarray(vaa_usr, float)
    if vza.ndim == 1 and vaa_.ndim == 1 and len(vza) != len(vaa_):
        VZ, VA = np.meshgrid(vza, vaa_, indexing="ij")
        vza = VZ.ravel()
        vaa_ = VA.ravel()

    # ---- slope preprocessing (test.f: apply once, before the rounds) ----
    H, cover, hotspot = effective_canopy_np(H, cover, hotspot, slope, tree_type)
    sza_t, saa_t, vza_t, vaa_t = slope_geometry_np(sza, vaa, vza, vaa_, slope, aspect)
    slope_rad = np.deg2rad(slope)
    al0 = slope_rad if alpha_slope is None else alpha_slope

    zcel = np.linspace(0.0, H, nz + 1)
    p = p_profile(zcel, H, cover, tree_type)
    h = np.diff(zcel)
    mu0 = np.abs(np.cos(np.deg2rad(sza_t)))
    fi0_hot, fi0 = fortran_fi0(saa_t)      # Fortran real(4) azimuth roundtrip

    if quad is None:
        b, fi, w0 = carlson_quadrature(sza_t)
    else:
        b, fi, w0 = quad
    b_usr, fi_usr = user_directions_slope(vza, vaa_, slope, aspect)

    K, kkH = kernel_arrays(zcel, b, H, cover, fi=fi, alpha=slope_rad,
                           diam=diam, tree_type=tree_type, alpha_leaf=al0)
    Kusr, kkHusr = kernel_arrays(zcel, b_usr, H, cover, fi=fi_usr, alpha=slope_rad,
                                 diam=diam, tree_type=tree_type, alpha_leaf=al0)
    K0, kkH0 = kernel_sun(zcel, mu0, H, cover, fi0=fi0 - np.pi, alpha=slope_rad,
                          diam=diam, tree_type=tree_type, alpha_leaf=al0)
    Kd0 = np.diag(K0)

    SCar = scatter_matrices(b, fi, b, fi, ro, tau, dL, lad, al0)
    SUsr = scatter_matrices(b, fi, b_usr, fi_usr, ro, tau, dL, lad, al0)
    SCar0D, SCar0U = scatter_sun(b, fi, mu0, fi0, ro, tau, dL, lad, al0)
    SUsr0D, SUsr0U = scatter_sun(b_usr, fi_usr, mu0, fi0, ro, tau, dL, lad, al0)

    al = slope_rad if alpha_slope is None else alpha_slope
    g_sun = float(np.ravel(ladmod.gfun(lad, -mu0, fi0, al))[0])
    g_dir_down = ladmod.gfun(lad, -b, fi + np.pi, al)
    g_dir_up = ladmod.gfun(lad, b, fi, al)
    g_usr_down = ladmod.gfun(lad, -b_usr, fi_usr + np.pi, al)
    g_usr_up = ladmod.gfun(lad, b_usr, fi_usr, al)
    sig0 = dL * g_sun
    lam0 = sig0 / mu0
    if problem == "BS_dir":
        ht_u = hotspot_factor(hotspot, dL, -mu0, fi0_hot, b, fi, sig0, dL * g_dir_up)
        sigCarD = dL * g_dir_down
        sigCarU = dL * g_dir_up * ht_u
        ht_u_usr = hotspot_factor(hotspot, dL, -mu0, fi0_hot, b_usr, fi_usr,
                                  sig0, dL * g_usr_up)
        sigUsrD = dL * g_usr_down
        sigUsrU = dL * g_usr_up * ht_u_usr
        i0 = 1.0
        d0 = np.zeros_like(b)
        dH = np.zeros_like(b)
        d0_usr = np.zeros_like(b_usr)
        dH_usr = np.zeros_like(b_usr)
    elif problem == "BS_dif":
        sigCarD = dL * g_dir_down
        sigCarU = dL * g_dir_up
        sigUsrD = dL * g_usr_down
        sigUsrU = dL * g_usr_up
        i0 = 0.0
        d0 = np.full_like(b, 1.0 / np.pi)
        dH = np.zeros_like(b)
        d0_usr = np.full_like(b_usr, 1.0 / np.pi)
        dH_usr = np.zeros_like(b_usr)
    elif problem == "S":
        sigCarD = dL * g_dir_down
        sigCarU = dL * g_dir_up
        sigUsrD = dL * g_usr_down
        sigUsrU = dL * g_usr_up
        i0 = 0.0
        d0 = np.zeros_like(b)
        dH = np.full_like(b, 1.0 / np.pi)
        d0_usr = np.zeros_like(b_usr)
        dH_usr = np.full_like(b_usr, 1.0 / np.pi)
    else:
        raise ValueError(problem)

    lamCarD = sigCarD / b
    lamCarU = sigCarU / b
    lamUsrD = sigUsrD / b_usr
    lamUsrU = sigUsrU / b_usr

    # Step 1: direct radiation
    one = np.ones((nz + 1, 1))
    udir, wzu, psi0, wzD = volterra(zcel, np.array([lam0]), np.array([lam0]),
                                    one, one, K0[None], kkH0[None])
    udir = udir[:, 0]
    psi0 = psi0[:, 0]
    idir = get_idir(zcel, p, lam0, Kd0, psi0, udir)

    # Step 2: internal sources
    fD0, fU0, fiD0, fiU0 = get_f0_fi0(zcel, p, K, kkH, lam0, SCar0D, SCar0U,
                                      udir, idir, psi0, d0, dH, i0, b)
    fUsrD0, fUsrU0, fiUsrD0, fiUsrU0 = get_f0_fi0(
        zcel, p, Kusr, kkHusr, lam0, SUsr0D, SUsr0U, udir, idir, psi0,
        d0_usr, dH_usr, i0, b_usr)

    # Step 3: initial approximation in Carlson directions
    vDk, vUk, psi, psiH = volterra(zcel, lamCarD, lamCarU, fD0, fU0, K, kkH)

    # Step 4: iterations
    STCarD = np.zeros((nz + 1, b.shape[0]))
    STCarU = np.zeros((nz + 1, b.shape[0]))
    fDk = fUk = None
    epsv0 = norm_volterra(zcel, p, w0, lamCarD, lamCarU, sigCarD, sigCarU,
                          K, kkH, vDk, vUk, psi, psiH)
    norm = epsv0
    reldel = 1.0
    it = 6
    n_iter = 0
    while reldel > 1e-7 and it <= maxiter:
        fDk, fUk = get_fk(zcel, b, K, kkH, SCar, w0, lamCarD, lamCarU,
                          K, kkH, vDk, vUk, psi, psiH)
        vDk, vUk, psi, psiH = volterra(zcel, lamCarD, lamCarU, fDk, fUk, K, kkH)
        STCarD += fDk
        STCarU += fUk
        epsv1 = norm_volterra(zcel, p, w0, lamCarD, lamCarU, sigCarD, sigCarU,
                              K, kkH, vDk, vUk, psi, psiH)
        norm += epsv1
        reldel = epsv1 / norm if norm > 0 else 0.0
        epsv0 = epsv1
        it += 1
        n_iter += 1

    # Step 5: final iteration
    STCarD_iter = STCarD.copy()
    STCarU_iter = STCarU.copy()
    STCarD = STCarD + fD0
    STCarU = STCarU + fU0
    udifD, udifU, psi, psiH = volterra(zcel, lamCarD, lamCarU, STCarD, STCarU,
                                       K, kkH)
    ssD, ssU = get_intpS(zcel, p, b, SCar, w0, lamCarD, lamCarU, K, kkH,
                         udifD, udifU, psi, psiH)
    IdifD, IdifU = get_idif(zcel, p, lamCarD, lamCarU, fiD0, fiU0, ssD, ssU,
                            udifD, udifU, psi, psiH, K, kkH)

    # user directions
    STUsrD, STUsrU = get_fk(zcel, b_usr, Kusr, kkHusr, SUsr, w0,
                            lamCarD, lamCarU, K, kkH,
                            udifD, udifU, psi, psiH)
    STUsrD_fk = STUsrD.copy()
    STUsrU_fk = STUsrU.copy()
    STUsrD = STUsrD + fUsrD0
    STUsrU = STUsrU + fUsrU0
    udifUsrD, udifUsrU, psiusr, psiHusr = volterra(
        zcel, lamUsrD, lamUsrU, STUsrD, STUsrU, Kusr, kkHusr)
    ssUsrD, ssUsrU = get_intpS(zcel, p, b_usr, SUsr, w0, lamCarD, lamCarU,
                               K, kkH, udifD, udifU, psi, psiH)
    IdifUsrD, IdifUsrU = get_idif(zcel, p, lamUsrD, lamUsrU, fiUsrD0, fiUsrU0,
                                  ssUsrD, ssUsrU, udifUsrD, udifUsrU,
                                  psiusr, psiHusr, Kusr, kkHusr)

    flux_iD, flux_iU = get_h_fluxes(b, w0, IdifD, IdifU)
    Tdir = i0 * mu0 * idir[-1]
    Adir = (1.0 - ro - tau) * mu0 * i0 * (1.0 - idir[-1])
    return {
        "zcel": zcel, "p": p, "b": b, "fi": fi, "w0": w0,
        "b_usr": b_usr, "fi_usr": fi_usr,
        "K": K, "kkH": kkH, "Kusr": Kusr, "kkHusr": kkHusr,
        "K0": K0, "kkH0": kkH0,
        "SCar": SCar, "SUsr": SUsr, "SCar0D": SCar0D, "SCar0U": SCar0U,
        "SUsr0D": SUsr0D, "SUsr0U": SUsr0U,
        "sigCarD": sigCarD, "sigCarU": sigCarU,
        "lamCarD": lamCarD, "lamCarU": lamCarU,
        "sigUsrD": sigUsrD, "sigUsrU": sigUsrU,
        "lamUsrD": lamUsrD, "lamUsrU": lamUsrU,
        "udir": udir, "psi0": psi0, "idir": idir, "lam0": lam0,
        "fD0": fD0, "fU0": fU0, "fiD0": fiD0, "fiU0": fiU0,
        "fUsrD0": fUsrD0, "fUsrU0": fUsrU0,
        "fiUsrD0": fiUsrD0, "fiUsrU0": fiUsrU0,
        "vDk_init": None, "STCarD": STCarD_iter, "STCarU": STCarU_iter,
        "STCarD_final": STCarD, "STCarU_final": STCarU,
        "fDk_last": fDk, "fUk_last": fUk,
        "udifD": udifD, "udifU": udifU, "psi": psi, "psiH": psiH,
        "ssD": ssD, "ssU": ssU, "IdifD": IdifD, "IdifU": IdifU,
        "STUsrD": STUsrD_fk, "STUsrU": STUsrU_fk,
        "STUsrD_final": STUsrD, "STUsrU_final": STUsrU,
        "udifUsrD": udifUsrD, "udifUsrU": udifUsrU,
        "psiusr": psiusr, "psiHusr": psiHusr,
        "ssUsrD": ssUsrD, "ssUsrU": ssUsrU,
        "IdifUsrD": IdifUsrD, "IdifUsrU": IdifUsrU,
        "flux_iD": flux_iD, "flux_iU": flux_iU,
        "Tdir": Tdir, "Adir": Adir, "n_iter": n_iter,
        "mu0": mu0, "sza": sza, "vaa": vaa,
        "sza_t": sza_t, "saa_t": saa_t, "vza_t": vza_t, "vaa_t": vaa_t,
        "slope": slope, "aspect": aspect, "lad": lad,
        "H_eff": H, "cover_eff": cover, "hotspot_eff": hotspot,
        "brf_plane_factor": mu0 / max(np.cos(np.deg2rad(sza)), 1e-9) if problem == "BS_dir" else 1.0,
    }

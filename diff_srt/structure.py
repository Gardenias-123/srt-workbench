"""Canopy structure functions used by the minimal SRT prototype.

The formulas for p(z) follow the Fortran routine ``get_probability`` in
``SRT/getPandKvalue.f``.  z is measured downward from the canopy top:
z=0 at the top, z=H at the bottom.
"""

import jax.numpy as jnp


def p_profile(z, H, cover, tree_type=4):
    """Vegetation probability p(z).

    Parameters
    ----------
    z : array_like
        Depth below canopy top (m).
    H : float
        Canopy height (m).
    cover : float
        Vegetation cover parameter in (0, 1); same role as ``VegCover``
        in the Fortran input file.
    tree_type : int
        1 cylinder, 2 cone, 3 ellipsoid, 4 sphere.
    """
    u = jnp.clip(z / H, 0.0, 1.0)
    if tree_type == 1:
        return cover * jnp.ones_like(z)
    if tree_type == 2:
        return 1.0 - (1.0 - cover) ** (u**2)
    if tree_type in (3, 4):
        return 1.0 - (1.0 - cover) ** (4.0 * u * (1.0 - u))
    return cover * jnp.ones_like(z)


def _g_numeric(mu, az, lad_type, alpha_slope):
    """Numerical G-function for LAD types 1-5, matching get_G_fun_v07.f."""
    a, b = {
        1: (1.0, 2.0),    # planophile
        2: (-1.0, 2.0),   # erectophile
        3: (-1.0, 4.0),   # plagiophile
        4: (1.0, 4.0),    # extremophile
        5: (0.0, 2.0),    # uniform
    }[lad_type]
    n, m = 30, 120
    h_theta = 0.5 * jnp.pi / n
    h_fi = 2.0 * jnp.pi / m
    theta = (jnp.arange(n) + 0.5) * h_theta
    phi = (jnp.arange(m) + 0.5) * h_fi
    th = theta[:, None]
    ph = phi[None, :]
    z = -jnp.sin(alpha_slope) * jnp.sin(th) * jnp.cos(ph) + jnp.cos(alpha_slope) * jnp.cos(th)
    theta_tran = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    theta_tran = jnp.where(theta_tran > jnp.pi / 2.0, jnp.pi - theta_tran, theta_tran)
    weight = 1.0 + a * jnp.cos(b * theta_tran)

    mu_arr = jnp.atleast_1d(mu)
    az_arr = jnp.atleast_1d(az)
    sin_mu = jnp.sqrt(jnp.maximum(1.0 - mu_arr**2, 1e-12))
    dot = jnp.abs(
        mu_arr[:, None, None] * jnp.cos(th)[None, :, :]
        + sin_mu[:, None, None] * jnp.sin(th)[None, :, :] * jnp.cos(az_arr[:, None, None] - ph[None, :, :])
    )
    sum1 = jnp.sum(dot * weight[None, :, :], axis=2)
    g = jnp.sum(sum1, axis=1) * h_fi * h_theta / (jnp.pi**2)
    return g[0] if jnp.ndim(mu) == 0 and jnp.ndim(az) == 0 else g


def g_function(mu, lad_type=6, az=0.0, alpha_slope=0.0):
    """G-function for leaf angle distributions.

    lad_type follows the Fortran convention:
    1 planophile, 2 erectophile, 3 plagiophile, 4 extremophile,
    5 uniform, 6 spherical, 7 horizontal, 8 vertical, 9 atmosphere.
    For 1-5 the numerical integral includes the geotropic LAD rotation
    by ``alpha_slope`` (radians), matching ``get_G_fun_v07.f``.
    """
    mu_abs = jnp.abs(mu)
    if lad_type == 6:
        return 0.5 + 0.0 * mu_abs
    if lad_type == 7:
        return mu_abs
    if lad_type == 8:
        return 0.6366197 * jnp.sqrt(jnp.maximum(1.0 - mu_abs**2, 0.0))
    if lad_type == 9:
        return 1.0 + 0.0 * mu_abs
    if 1 <= lad_type <= 5:
        return _g_numeric(mu_abs, az, lad_type, alpha_slope)
    raise NotImplementedError(f"LAD type {lad_type} not implemented")


def extinction(d_L, mu, lad_type=6, az=0.0, alpha_slope=0.0):
    """Volume extinction coefficient sigma = d_L * G(mu)."""
    return d_L * g_function(mu, lad_type, az, alpha_slope)


def hotspot_factor(mu_s, az_s, mu_v, az_v, hotspot, sig_s, sig_v):
    """Kuusk hot-spot correction for the extinction coefficient.

    This mirrors ``get_HSF`` in ``SRT_topo/get__HSF_AKUUSK_v06.f``.
    ``hotspot`` is H/S_L (same as the Fortran input); a value <= 0 means
    no correction.
    """
    if hotspot is None or float(hotspot) <= 0.0:
        return 1.0
    mu_s = jnp.abs(mu_s)
    mu_v = jnp.abs(mu_v)
    sin_s = jnp.sqrt(jnp.maximum(1.0 - mu_s**2, 0.0))
    sin_v = jnp.sqrt(jnp.maximum(1.0 - mu_v**2, 0.0))
    cosg = mu_s * mu_v + sin_s * sin_v * jnp.cos(az_v - az_s)
    delta = jnp.sqrt(jnp.abs(mu_s**2 + mu_v**2 + 2.0 * cosg * jnp.abs(mu_s * mu_v))) / jnp.maximum(
        jnp.abs(mu_s * mu_v), 1e-6
    )
    x = hotspot * delta
    # (1-exp(-x))/x with a stable limit at x -> 0
    ratio = (1.0 - jnp.exp(-x)) / jnp.maximum(x, 1e-6)
    fac = 1.0 - jnp.sqrt(
        jnp.abs(sig_s * mu_v) / jnp.maximum(jnp.abs(sig_v * mu_s), 1e-9)
    ) * ratio
    return jnp.clip(jnp.abs(fac), 0.0, 1.0)


def cumtrapz(y, z):
    """Cumulative trapezoidal integral, starting at 0."""
    seg = 0.5 * (y[:-1] + y[1:]) * jnp.diff(z)
    return jnp.concatenate([jnp.zeros((1,), dtype=y.dtype), jnp.cumsum(seg)])


def trapz(y, z):
    """Trapezoidal integral."""
    return jnp.sum(0.5 * (y[:-1] + y[1:]) * jnp.diff(z))


def lai(H, cover, d_L, tree_type=4, nz=400):
    """Leaf area index derived from the structural parameters.

    LAI = integral d_L * p(z) dz.
    """
    z = jnp.linspace(0.0, H, nz + 1)
    p = p_profile(z, H, cover, tree_type)
    return d_L * trapz(p, z)


def k_sphere(z1, z2, mu, H, cover):
    """Conditional pair-correlation K(z1, z2, mu) for spherical crowns.

    This is a direct translation of the ``case (4)`` branch of
    ``cond_pcf`` in ``SRT/getPandKvalue.f``.  Scalar inputs only.
    """
    eps = 1e-6
    pz1 = p_profile(z1, H, cover, tree_type=4)
    pz2 = p_profile(z2, H, cover, tree_type=4)

    epsz = 1e-3 * H
    lam = jnp.abs(z1 - z2) * jnp.sqrt(1.0 - mu**2) / jnp.maximum(0.0017, jnp.abs(mu))
    rz1 = jnp.sqrt(jnp.maximum(z1 * jnp.abs(H - z1), eps))
    rz2 = jnp.sqrt(jnp.maximum(z2 * jnp.abs(H - z2), eps))

    # Case: z1 on a boundary.
    cpf_z1_bound = jnp.where(lam <= rz2, 1.0, pz2)

    # Case: z2 on a boundary.
    cpf_z2_bound = 0.0

    # General interior case.
    xrz = jnp.minimum(rz1, rz2)
    xrs = jnp.maximum(rz1, rz2)

    kappa_far = 1.0 - (xrz / jnp.maximum(rz1, eps)) ** 2
    kappa_no_overlap = 1.0
    kappa_partial = _kappa_partial(lam, xrz, xrs, rz1)
    kappa = jnp.where(
        xrs >= xrz + lam,
        kappa_far,
        jnp.where(lam >= xrz + xrs, kappa_no_overlap, kappa_partial),
    )

    cpf = (1.0 - pz1) ** kappa
    cpf = (1.0 - cpf) / jnp.maximum(pz1, eps)
    cpf = 1.0 - (1.0 - pz2) * cpf

    cpf = jnp.where(
        (z1 <= epsz) | (jnp.abs(H - z1) <= epsz),
        cpf_z1_bound,
        jnp.where((z2 <= epsz) | (jnp.abs(H - z2) <= epsz), cpf_z2_bound, cpf),
    )
    return jnp.clip(cpf, 0.0, 1.0)


def _kappa_partial(lam, xrz, xrs, rz1):
    """Overlap term for two circles (Fortran interior branch)."""
    eps = 1e-12
    # Keep arccos arguments away from +/-1 so that the derivative stays
    # finite; JAX evaluates both branches of jnp.where during tracing.
    lim = 1.0 - 1e-7

    alfa = 2.0 * lam * xrs
    alfa = (xrs**2 - xrz**2 + lam**2) / jnp.maximum(alfa, eps)
    alfa = jnp.arccos(jnp.clip(alfa, -lim, lim))

    beta = 2.0 * lam * xrz
    beta = (xrz**2 - xrs**2 + lam**2) / jnp.maximum(beta, eps)
    beta = jnp.arccos(jnp.clip(beta, -lim, lim))

    kappa = alfa * xrs**2 + beta * xrz**2 - lam * xrs * jnp.sin(alfa)
    kappa = kappa / (jnp.pi * jnp.maximum(rz1**2, eps))
    return 1.0 - kappa


def k_cylinder(z1, z2, mu, H, cover, diam, alpha_slope=0.0, az=0.0):
    """Conditional PCF for cylindrical crowns with slope stretching.

    Direct translation of the ``case (1)`` branch of ``cond_pcf`` in
    ``SRT_topo/getPandKvalue.f``.  Scalar inputs only.
    """
    eps = 1e-9
    epsz = 1e-3 * H
    epsmu = 0.0017
    mu_abs = jnp.abs(mu)
    sin_mu = jnp.sqrt(jnp.maximum(1.0 - mu_abs**2, eps))

    lam = jnp.where(
        mu_abs >= 1.0 - 1e-7,
        0.0,
        jnp.abs(z1 - z2) * sin_mu / jnp.maximum(epsmu, mu_abs),
    )
    m = jnp.sin(az)
    n = jnp.cos(az) * jnp.cos(alpha_slope) + mu_abs * jnp.sin(alpha_slope) / jnp.maximum(sin_mu, eps)
    lam = lam * jnp.sqrt(m * m + n * n)
    R = diam / 2.0

    # Boundary cases use the physical distance.
    cpf_z1 = jnp.where(lam <= R, 1.0, cover)
    cpf_z2 = 0.0

    lam_n = lam / jnp.maximum(diam, eps)
    theta = jnp.arccos(jnp.clip(lam_n, -1.0, 1.0))
    kappa = 2.0 - (2.0 / jnp.pi) * (theta - lam_n * jnp.sqrt(jnp.maximum(1.0 - lam_n**2, 0.0)))
    cpf_inside = (2.0 * cover - 1.0 + (1.0 - cover) ** kappa) / jnp.maximum(cover, eps)
    cpf = jnp.where(lam_n < 1.0, cpf_inside, cover)

    cpf = jnp.where(z1 <= epsz, cpf_z1, jnp.where(z2 <= epsz, cpf_z2, cpf))
    return jnp.clip(cpf, 0.0, 1.0)

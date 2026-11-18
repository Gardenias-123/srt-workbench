"""Minimal differentiable forward model: direct beam + single scattering.

This is intentionally a reduced model.  It keeps the structural p(z) and
the extinction coefficient, but ignores diffuse multiple scattering and
the conditional pair-correlation K in the radiative transfer solve.
The purpose is to validate the AD pipeline and gradient checks before
porting the full SRT solver.
"""

import jax
import jax.numpy as jnp

from .solver import direct_beam
from .structure import cumtrapz, g_function, hotspot_factor, p_profile, trapz
from .topography import (
    brf_plane_from_slope,
    deg2rad,
    effective_canopy,
    plane_to_slope_az,
    plane_to_slope_mu,
)


def brf_single_scattering(
    H,
    cover,
    d_L,
    mu_s,
    mu_v,
    ro_soil,
    omega,
    tree_type=4,
    lad_type=6,
    nz=200,
):
    """Directional BRF for a flat canopy.

    Parameters
    ----------
    H : float
        Canopy height (m).
    cover : float
        Vegetation cover parameter.
    d_L : float
        Leaf area volume density (m2/m3).
    mu_s, mu_v : float
        Cosine of solar / view zenith angle (positive).
    ro_soil : float
        Soil reflectance.
    omega : float
        Leaf single-scattering albedo (ro_L + tau_L).
    tree_type : int
        Crown shape passed to ``p_profile``.
    lad_type : int
        Leaf angle distribution passed to ``g_function``.
    nz : int
        Number of vertical layers.

    Returns
    -------
    float
        BRF at the given sun/view geometry.
    """
    z = jnp.linspace(0.0, H, nz + 1)
    p = p_profile(z, H, cover, tree_type)

    # Optical depth from the top to depth z, per unit cosine.
    int_p = cumtrapz(p, z)

    sig_s = d_L * g_function(mu_s, lad_type)
    sig_v = d_L * g_function(mu_v, lad_type)

    T_s = jnp.exp(-sig_s / mu_s * int_p)
    T_v = jnp.exp(-sig_v / mu_v * int_p)

    # Soil contribution: direct beam reaches the soil, then is attenuated
    # along the upward view path.
    brf_soil = ro_soil * T_s[-1] * T_v[-1]

    # Leaf single scattering: isotropic phase function 1/(4*pi).
    brf_leaf = (omega * sig_s) / (4.0 * mu_s * mu_v) * trapz(p * T_s * T_v, z)

    return brf_soil + brf_leaf


def brf_topo(
    H,
    cover,
    d_L,
    tree_type,
    lad_type,
    diam,
    alpha_slope,
    sza,
    saa,
    vza,
    vaa,
    slope,
    aspect,
    ro_soil,
    ro_L,
    tau_L,
    nz=40,
    hotspot=0.0,
):
    """Slope-aware differentiable BRF for an array of view directions.

    The direct beam uses the K-based Volterra solution.  The upward
    single-scattering path still uses a Beer-law attenuation with p(z),
    so this is a reduced model (no diffuse multiple scattering).  It is
    intended for gradient-based inversion and sensitivity analysis, not
    as a replacement for the full Fortran forward model.
    """
    vza = jnp.atleast_1d(vza)
    vaa = jnp.atleast_1d(vaa)
    if vza.shape != vaa.shape:
        vza = jnp.repeat(vza, vaa.size)
        vaa = jnp.tile(vaa, vza.size // vaa.size)

    H_eff, cover_eff, _ = effective_canopy(H, cover, 0.0, slope, tree_type)
    slope_rad = deg2rad(slope)
    aspect_rad = deg2rad(aspect)

    # Use mu = cos(theta) directly to avoid the coordinate singularity of
    # arccos at the slope normal.
    mu_s = plane_to_slope_mu(deg2rad(sza), deg2rad(saa), slope_rad, aspect_rad)
    mu_v = plane_to_slope_mu(deg2rad(vza), deg2rad(vaa), slope_rad, aspect_rad)
    if lad_type == 6:
        az_s = 0.0
        az_v = jnp.zeros_like(mu_v)
    else:
        az_s = plane_to_slope_az(deg2rad(sza), deg2rad(saa), slope_rad, aspect_rad)
        az_v = plane_to_slope_az(deg2rad(vza), deg2rad(vaa), slope_rad, aspect_rad)

    z = jnp.linspace(0.0, H_eff, nz + 1)
    p = p_profile(z, H_eff, cover_eff, tree_type)
    U_dir, I_dir = direct_beam(
        z,
        H_eff,
        cover_eff,
        d_L,
        mu_s,
        tree_type=tree_type,
        lad_type=lad_type,
        diam=diam,
        alpha_slope=alpha_slope,
        az=az_s,
        hotspot=hotspot,
    )
    omega = ro_L + tau_L
    sig_s = d_L * g_function(mu_s, lad_type, az_s, alpha_slope)
    int_p = cumtrapz(p, z)

    def one(mu_v_, az_v_):
        sig_v = d_L * g_function(mu_v_, lad_type, az_v_, alpha_slope)
        h_v = hotspot_factor(mu_s, az_s, mu_v_, az_v_, hotspot, sig_s, sig_v)
        sig_v_h = sig_v * h_v
        T_v = jnp.exp(-sig_v_h / mu_v_ * int_p)
        leaf = omega * sig_s / (4.0 * mu_s * mu_v_) * trapz(p * U_dir * T_v, z)
        soil = ro_soil * I_dir[-1] * T_v[-1]
        return leaf + soil

    brf_slope = jax.vmap(one)(mu_v, az_v)
    brf_plane = brf_plane_from_slope(brf_slope, sza, mu_s)
    return brf_plane, brf_slope

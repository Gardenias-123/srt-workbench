"""TopoSRT coordinate transforms and slope-frame structural rules."""

from __future__ import annotations

import jax.numpy as jnp


def deg2rad(x):
    return x * jnp.pi / 180.0


def rad2deg(x):
    return x * 180.0 / jnp.pi


def _slope_components(theta, phi, slope, aspect):
    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)
    dphi = phi - aspect
    x = jnp.cos(dphi) * sin_t * jnp.cos(slope) - cos_t * jnp.sin(slope)
    y = jnp.sin(aspect - phi) * sin_t
    z = jnp.cos(dphi) * sin_t * jnp.sin(slope) + cos_t * jnp.cos(slope)
    return x, y, z


def plane_to_slope(theta, phi, slope, aspect):
    """Transform a direction from the plane frame to the slope frame.

    Parameters are in radians.  theta is the polar angle from the local
    zenith, phi the azimuth.  The formulas follow Eqs. (10.1)-(10.5) of
    ``SRT_理论完整推导.md``.
    """
    x, y, z = _slope_components(theta, phi, slope, aspect)
    theta_s = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    phi_s = jnp.arctan2(y, x)
    phi_s = jnp.pi / 2.0 - phi_s
    phi_s = jnp.mod(phi_s, 2.0 * jnp.pi)
    return theta_s, phi_s


def plane_to_slope_mu(theta, phi, slope, aspect):
    """Cosine of the slope-frame polar angle, without arccos.

    This is the differentiable path used by the forward model.  It has
    no coordinate singularity at the slope normal.
    """
    _, _, z = _slope_components(theta, phi, slope, aspect)
    return jnp.clip(z, -1.0, 1.0)


def plane_to_slope_az(theta, phi, slope, aspect):
    """Slope-frame azimuth in radians, with a safe denominator at nadir."""
    x, y, _ = _slope_components(theta, phi, slope, aspect)
    phi_s = jnp.arctan2(y, x + 1e-12)
    phi_s = jnp.pi / 2.0 - phi_s
    return jnp.mod(phi_s, 2.0 * jnp.pi)


def slope_geometry(sza, saa, vza, vaa, slope, aspect):
    """Local slope-frame angles for sun and view, returned in degrees."""
    sza_s, saa_s = plane_to_slope(deg2rad(sza), deg2rad(saa), deg2rad(slope), deg2rad(aspect))
    vza_s, vaa_s = plane_to_slope(deg2rad(vza), deg2rad(vaa), deg2rad(slope), deg2rad(aspect))
    return rad2deg(sza_s), rad2deg(saa_s), rad2deg(vza_s), rad2deg(vaa_s)


def effective_canopy(H, cover, hot_spot, slope, tree_type):
    """Apply the manuscript's slope rules to canopy structure.

    Spherical crowns: height unchanged, FVC scaled by cos(slope).
    Non-spherical crowns: FVC unchanged, effective height H*cos(slope).
    """
    cos_slope = jnp.cos(deg2rad(slope))
    is_sphere = jnp.asarray(tree_type) == 4
    H_eff = jnp.where(is_sphere, H, H * cos_slope)
    cover_eff = jnp.where(is_sphere, cover * cos_slope, cover)
    hot_eff = jnp.where(is_sphere, hot_spot, hot_spot * cos_slope)
    return H_eff, cover_eff, hot_eff


def brf_plane_from_slope(brf_slope, sza_plane, mu_slope):
    """Convert slope-frame BRF to plane-frame BRF, Eq. (10.11)."""
    return brf_slope * mu_slope / jnp.maximum(jnp.cos(deg2rad(sza_plane)), 1e-6)

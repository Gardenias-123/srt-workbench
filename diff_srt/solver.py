"""Direct-beam Volterra solver for the SRT equations (flat terrain).

Implements the direct-beam part of A10/A12:

    U_dir(z) + sigma/mu_s * int_0^z K(z,z') U_dir(z') dz' = 1
    I_dir(z) + sigma/mu_s * int_0^z p(z') U_dir(z') dz' = 1

The integrals are discretized with the trapezoidal rule on the same z grid.
"""

import jax
import jax.numpy as jnp

from .structure import g_function, hotspot_factor, k_cylinder, k_sphere, p_profile


def trapezoid_weight_matrix(z):
    """Matrix W such that (W @ y)[i] = trapezoid integral of y from z[0] to z[i]."""
    n = z.shape[0]
    dz = z[1] - z[0]
    W = jnp.tril(jnp.ones((n, n)), -1) * dz
    W = W.at[:, 0].set(0.5 * dz)
    W = W + jnp.eye(n) * 0.5 * dz
    # Integral from z[0] to z[0] is zero, not half a cell.
    W = W.at[0, 0].set(0.0)
    return W


def k_matrix_sphere(z, mu, H, cover):
    """K(z_i, z_j, mu) matrix for spherical crowns."""
    return jax.vmap(
        lambda z1: jax.vmap(lambda z2: k_sphere(z1, z2, mu, H, cover))(z)
    )(z)


def k_matrix(z, mu, H, cover, tree_type=4, diam=5.0, alpha_slope=0.0, az=0.0):
    """K(z_i, z_j, mu) matrix for the supported crown shapes.

    Sphere and cylinder use their exact Fortran branches.  Cone and
    ellipsoid currently fall back to the spherical kernel with the
    effective H/cover, which is an approximation.
    """
    if tree_type == 4:
        return k_matrix_sphere(z, mu, H, cover)
    if tree_type == 1:
        return jax.vmap(
            lambda z1: jax.vmap(
                lambda z2: k_cylinder(z1, z2, mu, H, cover, diam, alpha_slope, az)
            )(z)
        )(z)
    return k_matrix_sphere(z, mu, H, cover)


def direct_beam(
    z,
    H,
    cover,
    d_L,
    mu_s,
    tree_type=4,
    lad_type=6,
    diam=5.0,
    alpha_slope=0.0,
    az=0.0,
    hotspot=0.0,
):
    """Solve the direct-beam SRT equations on grid z.

    Returns
    -------
    U_dir : array
        Vegetation-region direct intensity.
    I_dir : array
        All-scene direct intensity.
    """
    p = p_profile(z, H, cover, tree_type)
    sigma = d_L * g_function(mu_s, lad_type, az, alpha_slope)
    h = hotspot_factor(mu_s, az, mu_s, az, hotspot, sigma, sigma)
    sigma_h = sigma * h
    K = k_matrix(z, mu_s, H, cover, tree_type, diam, alpha_slope, az)
    W = trapezoid_weight_matrix(z)
    A = jnp.eye(z.shape[0]) + sigma_h / mu_s * W * K
    U_dir = jnp.linalg.solve(A, jnp.ones_like(z))
    I_dir = 1.0 - sigma_h / mu_s * (W @ (p * U_dir))
    return U_dir, I_dir

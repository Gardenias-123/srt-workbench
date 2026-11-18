"""Differentiable SRT / TopoSRT models.

Two generations live here:

* ``ref_srt`` / ``jax_srt`` - the full validated solver (flat terrain,
  spherical crowns, spherical LAD).  It ports the Fortran mean-kernel
  Volterra + diffuse multiple-scattering chain line by line (including all
  four scattering terms ScatDD/DU/UD/UU) and matches the Fortran reference
  BRF to ~1e-5.  Use ``jax_srt.brf_flat_sphere``.
* ``forward`` / ``solver`` / ``structure`` - the early reduced model
  (direct beam + single scattering) still used for slope/other crown types.

``fortran_style`` is an intermediate, superseded experiment.
"""

from .structure import (
    p_profile,
    g_function,
    extinction,
    cumtrapz,
    trapz,
    lai,
    k_sphere,
)
from .forward import brf_single_scattering, brf_topo
from .solver import direct_beam, k_matrix, k_matrix_sphere, trapezoid_weight_matrix
from .structure import k_cylinder
from .topography import (
    brf_plane_from_slope,
    effective_canopy,
    plane_to_slope,
    slope_geometry,
)
from .jax_srt import brf_flat_sphere

__all__ = [
    "brf_flat_sphere",
    "p_profile",
    "g_function",
    "extinction",
    "cumtrapz",
    "trapz",
    "lai",
    "k_sphere",
    "k_cylinder",
    "brf_single_scattering",
    "brf_topo",
    "direct_beam",
    "k_matrix",
    "k_matrix_sphere",
    "trapezoid_weight_matrix",
    "plane_to_slope",
    "slope_geometry",
    "effective_canopy",
    "brf_plane_from_slope",
]

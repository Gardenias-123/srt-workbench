"""Carlson quadrature helpers.

The validated implementation lives in :mod:`diff_srt.ref_srt` (ported line by
line from ``EQN2PI`` / ``EQN2PI_C_k1_k2`` / ``trans_DS_to_Plane`` and checked
against ``SRT_diag/000output/diag_quad.txt``).  This module re-exports it so
existing imports keep working.

Historical note: earlier versions of this file used the wrong 1-based
denominator (``w - 2*i + 2`` with a 0-based ``i``) and stepped the azimuth
blocks by ``pi`` instead of ``pi/2`` (in the Fortran routine the local
variable ``pi`` is ``acos(-1)/2``).  Both are fixed in the validated port.
"""

from __future__ import annotations

import numpy as np

from .ref_srt import (
    carlson_quadrature,
    eqn2pi,
    eqn2pi_c_k1_k2 as eqn2pi_c,
    trans_ds_to_plane,
)


def nodes(sza_deg, nkw=18, k1=4):
    """Return flattened (b, fi, w0) nodes of the plane-geometry quadrature."""
    b, fi, w0 = carlson_quadrature(sza_deg, nkw=nkw, k1=k1)
    return np.asarray(b, float), np.asarray(fi, float), np.asarray(w0, float)


__all__ = ["eqn2pi", "eqn2pi_c", "trans_ds_to_plane", "carlson_quadrature", "nodes"]

"""Regression test for the slope-frame diffuse BRF normalization fix."""

import numpy as np

from app.engines.fortran import _combine_tables


def test_mixed_brf_uses_plane_frame_for_diffuse():
    dir_run = {"brf_table": [{"SZA": 45.0, "SZA_T": 30.0, "BRF": 0.6}]}
    dif_run = {"brf_table": [{"SZA": 45.0, "SZA_T": 30.0, "BRF": 0.4}]}
    out = _combine_tables(dir_run, dif_run, 0.5)
    factor = np.cos(np.deg2rad(30.0)) / np.cos(np.deg2rad(45.0))
    assert abs(out["brf_plane"][0] - (0.5 * 0.6 + 0.5 * 0.4)) < 1e-12
    assert abs(out["brf_slope"][0] - (0.5 * 0.6 / factor + 0.5 * 0.4)) < 1e-12


def test_pure_diffuse_has_no_direct_cosine_factor():
    dif_run = {"brf_table": [{"SZA": 45.0, "SZA_T": 30.0, "BRF": 0.4}]}
    out = _combine_tables(None, dif_run, 0.0)
    assert abs(out["brf_plane"][0] - 0.4) < 1e-12
    assert abs(out["brf_slope"][0] - 0.4) < 1e-12

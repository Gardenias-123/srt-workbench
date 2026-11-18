"""Validation tests for the full differentiable JAX SRT solver.

The reference values come from the Fortran TopoSRT run of the standard case
(SZA=45, SAA=270, sphere crowns, H=5, cover=0.49397, nz=20).  The Fortran
output is single precision, so the tolerances below are at the 1e-5 level.
"""
from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diff_srt import jax_srt as js  # noqa: E402
from diff_srt import ref_srt as ref  # noqa: E402

DATA = Path(__file__).parent / "data"
PARAMS = dict(H=5.0, cover=0.49397, dL=1.0, ro=0.4746, tau=0.4843,
              hotspot=8.0, sza=45.0, saa=270.0, ro_soil=0.3379)


def _load(name):
    d = np.loadtxt(DATA / name, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2]


def test_carlson_quadrature_matches_fortran():
    b, fi, w0 = ref.carlson_quadrature(45.0)
    assert len(b) == 217 and len(fi) == 217 and len(w0) == 217
    assert np.isclose(w0.sum(), 2.0 * np.pi, rtol=1e-12)
    # first nodes from SRT_diag/000output/diag_quad.txt
    assert np.isclose(b[0], 0.0107059441507, atol=1e-6)
    assert np.isclose(fi[0], 1.43969821930, atol=1e-5)
    assert np.isclose(b[1], 0.130715489388, atol=1e-6)
    assert np.isclose(w0[0], 0.0349065884948, atol=1e-9)


def test_brf_matches_fortran_direct():
    vza, vaa, brf_ref = _load("fortran_brf_base_i0_1.csv")
    brf = js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                             PARAMS["sza"], PARAMS["saa"], vza, vaa,
                             PARAMS["ro"], PARAMS["tau"], PARAMS["ro_soil"],
                             hotspot=PARAMS["hotspot"], nz=20,
                             direct_fraction=1.0)
    brf = np.asarray(brf)
    assert np.all(np.isfinite(brf))
    assert np.abs(brf - brf_ref).max() < 3e-5
    assert abs(brf[0] - 0.38480) < 3e-5


def test_brf_matches_fortran_diffuse():
    vza, vaa, brf_ref = _load("fortran_brf_base_i0_0.csv")
    brf = js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                             PARAMS["sza"], PARAMS["saa"], vza, vaa,
                             PARAMS["ro"], PARAMS["tau"], PARAMS["ro_soil"],
                             hotspot=PARAMS["hotspot"], nz=20,
                             direct_fraction=0.0)
    brf = np.asarray(brf)
    assert np.abs(brf - brf_ref).max() < 3e-5


def test_mixed_illumination_is_linear():
    vza = np.array([0.0, 15.0, 30.0, 45.0]); vaa = np.array([0.0, 90.0, 180.0, 270.0])
    common = dict(ro=PARAMS["ro"], tau=PARAMS["tau"], ro_soil=PARAMS["ro_soil"],
                  hotspot=PARAMS["hotspot"], nz=10)
    b1 = np.asarray(js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                                       PARAMS["sza"], PARAMS["saa"], vza, vaa,
                                       direct_fraction=1.0, **common))
    b0 = np.asarray(js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                                       PARAMS["sza"], PARAMS["saa"], vza, vaa,
                                       direct_fraction=0.0, **common))
    bf = np.asarray(js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                                       PARAMS["sza"], PARAMS["saa"], vza, vaa,
                                       direct_fraction=0.35, **common))
    assert np.abs(bf - (0.35 * b1 + 0.65 * b0)).max() < 1e-10


@pytest.mark.skipif(not os.environ.get("SRT_SLOW_TESTS"),
                    reason="set SRT_SLOW_TESTS=1 to run the gradient compile test")
def test_gradients_finite_and_match_fd():
    vza = np.array([0.0, 30.0, 45.0, 60.0])
    vaa = np.array([0.0, 270.0, 90.0, 180.0])

    def loss(H, cover, dL):
        b = js.brf_flat_sphere(H, cover, dL, 45.0, 270.0, vza, vaa,
                               0.4746, 0.4843, 0.3379, hotspot=8.0, nz=6)
        return jnp.sum(b)

    grads = jax.grad(loss, argnums=(0, 1, 2))(3.0, 0.4, 0.8)
    assert all(bool(jnp.isfinite(g)) for g in grads)
    eps = 1e-4
    f0 = float(loss(3.0, 0.4, 0.8))
    fd_cover = (float(loss(3.0, 0.4 + eps, 0.8)) - f0) / eps
    assert abs(fd_cover - float(grads[1])) < 0.05 * max(abs(fd_cover), 1.0)


@pytest.mark.parametrize("name,slope,aspect", [
    ("fortran_brf_slope15_asp0.csv", 15.0, 0.0),
    ("fortran_brf_slope45_asp0.csv", 45.0, 0.0),
    ("fortran_brf_slope15_asp180.csv", 15.0, 180.0),
    ("fortran_brf_slope15_asp270.csv", 15.0, 270.0),
])
def test_brf_matches_fortran_slope(name, slope, aspect):
    vza, vaa, brf_ref = _load(name)
    brf = js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                             PARAMS["sza"], PARAMS["saa"], vza, vaa,
                             PARAMS["ro"], PARAMS["tau"], PARAMS["ro_soil"],
                             hotspot=PARAMS["hotspot"], nz=20,
                             slope=slope, aspect=aspect)
    brf = np.asarray(brf)
    assert np.all(np.isfinite(brf))
    assert np.abs(brf - brf_ref).max() < 5e-5


@pytest.mark.parametrize("name,slope,aspect", [
    ("fortran_brf_cyl_flat.csv", 0.0, 90.0),
    ("fortran_brf_cyl15_asp0.csv", 15.0, 0.0),
])
def test_brf_matches_fortran_cylinder(name, slope, aspect):
    vza, vaa, brf_ref = _load(name)
    brf = js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                             PARAMS["sza"], PARAMS["saa"], vza, vaa,
                             PARAMS["ro"], PARAMS["tau"], PARAMS["ro_soil"],
                             hotspot=PARAMS["hotspot"], nz=20,
                             slope=slope, aspect=aspect,
                             tree_type=1, diam=5.0)
    brf = np.asarray(brf)
    assert np.abs(brf - brf_ref).max() < 5e-5


def test_height_density_scale_invariance():
    """The SRT BRF is exactly invariant under H -> aH, d_L -> d_L/a."""
    vza = np.array([0.0, 30.0, 60.0]); vaa = np.array([90.0, 270.0, 270.0])
    kw = dict(sza=45.0, saa=270.0, ro=0.474672127, tau=0.484288576,
              ro_soil=0.3382, hotspot=8.0, nz=6, slope=15.0, aspect=270.0,
              tree_type=4, diam=5.0)
    a = np.asarray(js.brf_flat_sphere(5.0, 0.49, 1.0, vza=vza, vaa=vaa, **kw))
    b = np.asarray(js.brf_flat_sphere(10.0, 0.49, 0.5, vza=vza, vaa=vaa, **kw))
    assert np.abs(a - b).max() < 1e-10


def test_lai_parameterization_roundtrip():
    from diff_srt.inversion import dl_from_lai, lai_from_dl, shape_factor

    assert abs(float(shape_factor(0.49, 1)) - 0.49) < 1e-9
    for tree in (1, 4):
        for cover in (0.2, 0.49, 0.8):
            lai = lai_from_dl(1.3, 6.0, cover, tree)
            assert abs(float(dl_from_lai(lai, 6.0, cover, tree)) - 1.3) < 1e-5


@pytest.mark.skipif(not os.environ.get("SRT_SLOW_TESTS"),
                    reason="set SRT_SLOW_TESTS=1 to run the LAI inversion test")
def test_lai_inversion_identifiable():
    from diff_srt.inversion import (ModelConfig, Observation, lai_from_dl,
                                    predict, run_inversion)

    vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 0.0])
    vaa = np.array([270.0] * 5 + [90.0])
    model = ModelConfig(full_flat=True, slope=15.0, aspect=270.0, nz=6,
                        hotspot=8.0, ro_L=0.474672127, tau_L=0.484288576,
                        ro_soil=0.3382)
    x_true = jnp.asarray([5.0, 0.49, 1.0, 15.0, 270.0])
    obs = Observation(sza=45.0, saa=270.0, vza=vza, vaa=vaa,
                      brf=np.zeros(len(vza)), sigma=0.005)
    obs.brf = np.asarray(predict(x_true, obs, model))
    lai_true = float(lai_from_dl(1.0, 5.0, 0.49, 4))
    init = np.array([5.0, 0.62, 2.8, 15.0, 270.0])
    bounds = np.array([[5.0, 5.0], [0.05, 0.95], [0.1, 6.0], [15.0, 15.0], [270.0, 270.0]])
    out = run_inversion(obs, init, free_names=("cover", "LAI"), bounds=bounds,
                        model=model, n_iter=200, lr=0.05, prior_weight=1e-4)
    assert abs(out["free_retrieved"]["cover"] - 0.49) < 0.01
    assert abs(out["free_retrieved"]["LAI"] / lai_true - 1.0) < 0.02


@pytest.mark.parametrize("name,tree_type,lad", [
    ("fortran_brf_cone15.csv", 2, 6),
    ("fortran_brf_ell15.csv", 3, 6),
    ("fortran_brf_lad1_15.csv", 4, 1),
    ("fortran_brf_lad8_15.csv", 4, 8),
])
def test_brf_matches_fortran_crown_and_lad(name, tree_type, lad):
    vza, vaa, brf_ref = _load(name)
    brf = js.brf_flat_sphere(PARAMS["H"], PARAMS["cover"], PARAMS["dL"],
                             PARAMS["sza"], PARAMS["saa"], vza, vaa,
                             PARAMS["ro"], PARAMS["tau"], PARAMS["ro_soil"],
                             hotspot=PARAMS["hotspot"], nz=20,
                             slope=15.0, aspect=0.0,
                             tree_type=tree_type, diam=5.0, lad=lad)
    brf = np.asarray(brf)
    assert np.abs(brf - brf_ref).max() < 5e-5


@pytest.mark.skipif(not os.environ.get("SRT_SLOW_TESTS"),
                    reason="set SRT_SLOW_TESTS=1 to run the prior inversion test")
def test_h_prior_ties_height_to_prior():
    """With a Gaussian H prior, H's posterior equals the prior (zero sensitivity)."""
    from diff_srt.inversion import (ModelConfig, Observation, lai_from_dl,
                                    predict, run_inversion)

    vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
    vaa = np.full_like(vza, 270.0)
    model = ModelConfig(full_flat=True, slope=15.0, aspect=270.0, nz=6,
                        hotspot=8.0, ro_L=0.474672127, tau_L=0.484288576,
                        ro_soil=0.3382)
    x_true = jnp.asarray([5.0, 0.49, 1.0, 15.0, 270.0])
    obs = Observation(sza=45.0, saa=270.0, vza=vza, vaa=vaa,
                      brf=np.zeros(len(vza)), sigma=0.005)
    obs.brf = np.asarray(predict(x_true, obs, model))
    lai_true = float(lai_from_dl(1.0, 5.0, 0.49, 4))
    init = np.array([4.5, 0.6, 2.0, 15.0, 270.0])
    bounds = np.array([[1.0, 15.0], [0.05, 0.95], [0.1, 6.0], [15.0, 15.0], [270.0, 270.0]])
    out = run_inversion(
        obs, init, free_names=("H", "cover", "LAI"), bounds=bounds,
        model=model, n_iter=150, lr=0.05,
        prior_sigma=np.array([0.5, np.inf, np.inf, np.inf, np.inf]),
        prior_mean=np.array([5.5, 0.6, 2.0, 15.0, 270.0]),
    )
    # H is unconstrained by the data -> sits at the prior mean.
    assert abs(out["free_retrieved"]["H"] - 5.5) < 0.02
    assert abs(out["free_retrieved"]["cover"] - 0.49) < 0.01
    assert abs(out["free_retrieved"]["LAI"] / lai_true - 1.0) < 0.02
    assert abs(out["std"][out["free_names"].index("H")] - 0.5) < 0.02


@pytest.mark.skipif(not os.environ.get("SRT_SLOW_TESTS"),
                    reason="set SRT_SLOW_TESTS=1 to run the JIT-cache test")
def test_inversion_jit_cache_reuse():
    """A second inversion with new init/sigma/data must not recompile."""
    from diff_srt.inversion import (ModelConfig, Observation, _JIT_CACHE,
                                    clear_jit_cache, predict, run_inversion)

    vza = np.array([0.0, 30.0, 60.0, 0.0])
    vaa = np.array([270.0, 270.0, 270.0, 90.0])
    model = ModelConfig(full_flat=True, slope=15.0, aspect=270.0, nz=5,
                        hotspot=8.0, ro_L=0.474672127, tau_L=0.484288576,
                        ro_soil=0.3382)
    x_true = jnp.asarray([5.0, 0.49, 1.0, 15.0, 270.0])
    obs = Observation(sza=45.0, saa=270.0, vza=vza, vaa=vaa,
                      brf=np.zeros(len(vza)), sigma=0.005)
    obs.brf = np.asarray(predict(x_true, obs, model))
    bounds = np.array([[5.0, 5.0], [0.05, 0.95], [0.1, 6.0], [15.0, 15.0], [270.0, 270.0]])
    clear_jit_cache()
    run_inversion(obs, np.array([5.0, 0.6, 2.0, 15.0, 270.0]),
                  free_names=("cover", "LAI"), bounds=bounds, model=model,
                  n_iter=3, lr=0.05)
    n1 = len(_JIT_CACHE)
    obs2 = Observation(sza=45.0, saa=270.0, vza=vza, vaa=vaa,
                       brf=obs.brf * 1.01, sigma=0.01)
    run_inversion(obs2, np.array([5.0, 0.62, 2.2, 15.0, 270.0]),
                  free_names=("cover", "LAI"), bounds=bounds, model=model,
                  n_iter=3, lr=0.05)
    assert len(_JIT_CACHE) == n1


def test_engine_predict_uses_full_model():
    from app.params import SRTParams
    from app.engines.jax_engine import _forward_pair, _model
    from diff_srt.inversion import Observation, predict

    p = SRTParams()
    model = _model(p)
    assert model.full_flat
    zens, azis = p.view_grid()
    obs = Observation(sza=p.sza, saa=p.saa,
                      vza=np.asarray(zens), vaa=np.asarray(azis),
                      brf=np.zeros(len(zens)))
    pred = np.asarray(predict(jnp.asarray([p.H, p.cover, p.d_L, 0.0, p.aspect]),
                              obs, model))
    vza, vaa, brf_ref = _load("fortran_brf_base_i0_1.csv")
    assert np.abs(pred - brf_ref).max() < 3e-5
